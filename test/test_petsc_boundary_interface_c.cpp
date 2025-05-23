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

} // namespace

TEST(PETScBoundary2D, corners) {

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
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 2),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  ParticleSet initial_distribution(1, particle_spec);
  initial_distribution[Sym<REAL>("P")][0][0] = 0.5 * mesh_size * h;
  initial_distribution[Sym<REAL>("P")][0][1] = 0.5 * mesh_size * h;
  if (sycl_target->comm_pair.rank_parent == 0) {
    A->add_particles_local(initial_distribution);
  }
  A->hybrid_move();
  A->cell_move();

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);

  std::vector<std::array<REAL, 4>> offsets;
  offsets.push_back(
      {(mesh_size + 1) * h, (mesh_size + 1) * h, mesh_size * h, mesh_size * h});
  offsets.push_back({(mesh_size + 1) * h, -h, mesh_size * h, 0.0});
  offsets.push_back({-h, (mesh_size + 1) * h, 0.0, mesh_size * h});
  offsets.push_back({-h, -h, 0.0, 0.0});

  auto reset_loop = particle_loop(
      A,
      [=](auto P) {
        P.at(0) = mesh_size * 0.5 * h;
        P.at(1) = mesh_size * 0.5 * h;
      },
      Access::write(Sym<REAL>("P")));

  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();

  for (auto &ox : offsets) {
    REAL finalx = ox.at(0);
    REAL finaly = ox.at(1);
    REAL intersectx = ox.at(2);
    REAL intersecty = ox.at(3);

    reset_loop->execute();
    b2d->pre_integration(A);
    particle_loop(
        A,
        [=](auto P) {
          P.at(0) = finalx;
          P.at(1) = finaly;
        },
        Access::write(Sym<REAL>("P")))
        ->execute();
    auto sub_group = b2d->post_integration(A);
    if (A->get_npart_local() == 1) {
      ASSERT_EQ(sub_group.at(1)->get_npart_local(), 1);
    }

    particle_loop(
        sub_group.at(1),
        [=](auto B_P, auto B_C) {
          NESO_KERNEL_ASSERT(
              KERNEL_ABS(B_P.at_ephemeral(0) - intersectx) < 1.0e-15, k_ep);
          NESO_KERNEL_ASSERT(
              KERNEL_ABS(B_P.at_ephemeral(1) - intersecty) < 1.0e-15, k_ep);
          NESO_KERNEL_ASSERT(B_C.at_ephemeral(0) == 1, k_ep);
          NESO_KERNEL_ASSERT(B_C.at_ephemeral(1) > -1, k_ep);
        },
        Access::read(BoundaryInteractionSpecification::intersection_point),
        Access::read(BoundaryInteractionSpecification::intersection_metadata))
        ->execute();
    ASSERT_EQ(ep->get_flag(), 0);
  }

  b2d->free();
  sycl_target->free();
  A->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
