#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

namespace {
class TestBoundaryInteraction2D : public PetscInterface::BoundaryInteraction2D {
public:
  TestBoundaryInteraction2D(
      SYCLTargetSharedPtr sycl_target,
      PetscInterface::DMPlexInterfaceSharedPtr mesh,
      std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
      const REAL tol = 0.0,
      std::optional<Sym<REAL>> previous_position_sym = std::nullopt)
      :

        PetscInterface::BoundaryInteraction2D(
            sycl_target, mesh, boundary_groups, tol, previous_position_sym) {}

  using BoundaryInteractionCellData2D =
      PetscInterface::BoundaryInteractionCellData2D;
  using BoundaryInteractionNormalData2D =
      PetscInterface::BoundaryInteractionNormalData2D;

  MAKE_WRAP_METHOD(get_labels)
  MAKE_WRAP_METHOD(get_bounding_box)
  MAKE_WRAP_METHOD(collect_cells)
  MAKE_GETTER_METHOD(facets_real)
  MAKE_GETTER_METHOD(facets_int)
  MAKE_GETTER_METHOD(num_facets_global)
  MAKE_GETTER_METHOD(ncomp_int)
  MAKE_GETTER_METHOD(ncomp_real)
  MAKE_GETTER_METHOD(required_mh_cells)
  MAKE_GETTER_METHOD(collected_mh_cells)
  MAKE_GETTER_METHOD(d_map_edge_discovery)
  MAKE_GETTER_METHOD(d_map_edge_normals)
};

ParticleGroupSharedPtr particle_loop_common(DM dm, const int N = 1093) {

  const int ndim = 2;
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("U"), 3),
                             ParticleProp(Sym<REAL>("TSP"), 2),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 2),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  int ncell_local = mesh->get_cell_count();
  int ncell_global;

  MPICHK(MPI_Allreduce(&ncell_local, &ncell_global, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD));
  const int npart_per_cell = std::max(1, N / ncell_global);
  const int rank = sycl_target->comm_pair.rank_parent;
  const INT id_offset = rank * N;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  uniform_within_dmplex_cells(mesh, npart_per_cell, positions, cells, &rng_pos);

  const int N_actual = cells.size();
  auto velocities =
      NESO::Particles::normal_distribution(N_actual, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N_actual, particle_spec);

  for (int px = 0; px < N_actual; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px + id_offset;
  }

  A->add_particles_local(initial_distribution);

  return A;
}

} // namespace

TEST(PETScBoundary2D, pre_integrate) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 8;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces, lower, upper,
      /* periodicity */ NULL, PETSC_TRUE, &dm));
  PetscInterface::generic_distribute(&dm);
  auto A = particle_loop_common(dm, 128);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);
  // Test pre_integration
  b2d->pre_integration(A);
  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();
  particle_loop(
      A,
      [=](auto CURR_POS, auto PREV_POS) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          NESO_KERNEL_ASSERT(CURR_POS.at(dimx) == PREV_POS.at(dimx), k_ep);
        }
      },
      Access::read(A->position_dat), Access::read(b2d->previous_position_sym))
      ->execute();
  EXPECT_EQ(ep->get_flag(), 0);

  b2d->free();
  A->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary2D, post_integrate) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 8;
  const int npart_per_cell = 3;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces, lower, upper,
      /* periodicity */ NULL, PETSC_TRUE, &dm));
  PetscInterface::generic_distribute(&dm);
  auto A = particle_loop_common(dm, mesh_size * mesh_size * npart_per_cell);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2}; // x=L, y=0
  boundary_groups[2] = {3};    // y=L
  boundary_groups[3] = {4};    // x=0

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);
  // Test pre_integration
  b2d->pre_integration(A);
  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();
  particle_loop(
      A,
      [=](auto CURR_POS, auto PREV_POS, auto P2) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          NESO_KERNEL_ASSERT(CURR_POS.at(dimx) == PREV_POS.at(dimx), k_ep);
          P2.at(dimx) = CURR_POS.at(dimx);
        }
      },
      Access::read(A->position_dat), Access::read(b2d->previous_position_sym),
      Access::write(Sym<REAL>("P2")))
      ->execute();
  EXPECT_EQ(ep->get_flag(), 0);

  // Test post_integration
  auto reset_positions = particle_loop(
      A,
      [=](auto P, auto P2) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          P.at(dimx) = P2.at(dimx);
        }
      },
      Access::write(A->position_dat), Access::read(Sym<REAL>("P2")));

  // Offsets to move particles in +x, -x, +y, -y
  std::vector<
      std::tuple<std::array<REAL, 2>, int, REAL, std::array<REAL, 2>, int>>
      offsets;

  offsets.push_back({{h, 0}, 0, mesh_size * h, {1.0, 0.0}, 1});
  offsets.push_back({{-h, 0}, 0, 0.0, {-1.0, 0.0}, 3});
  offsets.push_back({{0, h}, 1, mesh_size * h, {0.0, 1.0}, 2});
  offsets.push_back({{0, -h}, 1, 0.0, {0.0, 1.0}, 1});

  for (auto &ox : offsets) {
    reset_positions->execute();
    const REAL dx = std::get<0>(ox).at(0);
    const REAL dy = std::get<0>(ox).at(1);
    const int component = std::get<1>(ox);
    const REAL value = std::get<2>(ox);
    const REAL normalx = std::get<3>(ox).at(0);
    const REAL normaly = std::get<3>(ox).at(1);
    const int correct_group = std::get<4>(ox);

    b2d->pre_integration(A);
    // move the particles in a direction
    particle_loop(
        A,
        [=](auto P) {
          P.at(0) += dx;
          P.at(1) += dy;
        },
        Access::write(A->position_dat))
        ->execute();
    auto sub_groups = b2d->post_integration(A);

    for (auto &sx : sub_groups) {
      ASSERT_TRUE(contains_boundary_interaction_data(sx.second, ndim));
      if (sx.first != correct_group) {
        ASSERT_EQ(sx.second->get_npart_local(), 0);
      }
    }

    // Check the boundary intersection is at the expected place
    particle_loop(
        sub_groups.at(correct_group),
        [=](auto BOUNDARY_POSITION) {
          NESO_KERNEL_ASSERT(
              KERNEL_ABS(BOUNDARY_POSITION.at_ephemeral(component) - value) <
                  1.0e-14,
              k_ep);
        },
        Access::read(BoundaryInteractionSpecification::intersection_point))
        ->execute();
    ASSERT_EQ(ep->get_flag(), 0);
    // Check the boundary metadata is as expected
    particle_loop(
        sub_groups.at(correct_group),
        [=](auto BOUNDARY_INFO) {
          NESO_KERNEL_ASSERT(BOUNDARY_INFO.at_ephemeral(0) == correct_group,
                             k_ep);
          NESO_KERNEL_ASSERT(BOUNDARY_INFO.at_ephemeral(1) > -1, k_ep);
        },
        Access::read(BoundaryInteractionSpecification::intersection_metadata))
        ->execute();
    ASSERT_EQ(ep->get_flag(), 0);
    // Check that the normal for the intersection point is as expected
    auto normal_mapper = b2d->get_device_normal_mapper();
    particle_loop(
        sub_groups.at(correct_group),
        [=](auto B_C) {
          REAL *normal;
          NESO_KERNEL_ASSERT(normal_mapper.get(B_C.at_ephemeral(1), &normal),
                             k_ep);

          const bool bx0 = KERNEL_ABS(normal[0] - normalx) < 1.0e-15;
          const bool bx1 = KERNEL_ABS(normal[0] + normalx) < 1.0e-15;
          const bool by0 = KERNEL_ABS(normal[1] - normaly) < 1.0e-15;
          const bool by1 = KERNEL_ABS(normal[1] + normaly) < 1.0e-15;

          NESO_KERNEL_ASSERT(bx0 || bx1, k_ep);
          NESO_KERNEL_ASSERT(by0 || by1, k_ep);
        },
        Access::read(BoundaryInteractionSpecification::intersection_metadata))
        ->execute();
    ASSERT_EQ(ep->get_flag(), 0);
  }

  b2d->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
