#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"

namespace {
class TestBoundaryInteraction2D : public PetscInterface::BoundaryInteraction2D {
public:
  TestBoundaryInteraction2D(
      SYCLTargetSharedPtr sycl_target,
      PetscInterface::DMPlexInterfaceSharedPtr mesh,
      std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
      const REAL tol = 0.0,
      std::optional<Sym<REAL>> previous_position_sym = std::nullopt,
      std::optional<Sym<REAL>> boundary_position_sym = std::nullopt,
      std::optional<Sym<INT>> boundary_label_sym = std::nullopt)
      :

        PetscInterface::BoundaryInteraction2D(
            sycl_target, mesh, boundary_groups, tol, previous_position_sym,
            boundary_position_sym, boundary_label_sym) {}

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

TEST(PETScBoundary2D, constructor_2d) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 32;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);

  auto labels = b2d->wrap_get_labels();
  ASSERT_TRUE(labels.count(1));
  ASSERT_TRUE(labels.count(2));
  ASSERT_TRUE(labels.count(3));
  ASSERT_TRUE(labels.count(4));

  const int num_facets_global = b2d->get_num_facets_global();
  const int ncomp_int = b2d->get_ncomp_int();
  const int ncomp_real = b2d->get_ncomp_real();
  REAL *ptr_real = b2d->get_facets_real();
  int *ptr_int = b2d->get_facets_int();

  auto lambda_find_int = [&](const int value) -> int {
    for (int ix = 0; ix < num_facets_global; ix++) {
      if (ptr_int[ix * ncomp_int + 1] == value) {
        return ix;
      }
    }
    return -1;
  };

  auto face_sets = mesh->dmh->get_face_sets();
  std::vector<std::vector<REAL>> coords;

  std::vector<REAL> test_real;
  std::vector<int> test_int;

  for (auto &item : face_sets) {
    const PetscInt face_id = item.first;
    if (labels.count(face_id)) {
      for (auto &point_index : item.second) {
        const PetscInt facet_global_id =
            mesh->dmh->get_point_global_index(point_index);

        auto index = lambda_find_int(facet_global_id);
        ASSERT_TRUE(index > -1);
        ASSERT_EQ(ptr_int[index * ncomp_int + 0], face_id);
        ASSERT_EQ(ptr_int[index * ncomp_int + 1], facet_global_id);

        test_int.push_back(face_id);
        test_int.push_back(facet_global_id);

        mesh->dmh->get_generic_vertices(point_index, coords);
        const REAL x0 = coords.at(0).at(0);
        const REAL y0 = coords.at(0).at(1);
        const REAL x1 = coords.at(1).at(0);
        const REAL y1 = coords.at(1).at(1);

        ASSERT_EQ(ptr_real[index * ncomp_real + 0], x0);
        ASSERT_EQ(ptr_real[index * ncomp_real + 1], y0);
        ASSERT_EQ(ptr_real[index * ncomp_real + 2], x1);
        ASSERT_EQ(ptr_real[index * ncomp_real + 3], y1);

        test_real.push_back(x0);
        test_real.push_back(y0);
        test_real.push_back(x1);
        test_real.push_back(y1);

        const REAL n0 = ptr_real[index * ncomp_real + 4];
        const REAL n1 = ptr_real[index * ncomp_real + 5];
        test_real.push_back(n0);
        test_real.push_back(n1);

        // Is the normal vector a unit normal vector
        ASSERT_NEAR(n0 * n0 + n1 * n1, 1.0, 1.0e-15);
        const REAL d0 = x1 - x0;
        const REAL d1 = y1 - y0;
        ASSERT_NEAR(d0 * n0 + d1 * n1, 0.0, 1.0e-15);
      }
    }
  }

  std::vector<REAL> test_global_real;
  std::vector<int> test_global_int;

  all_gather_v(test_real, MPI_COMM_WORLD, test_global_real);
  all_gather_v(test_int, MPI_COMM_WORLD, test_global_int);

  ASSERT_EQ(test_global_real.size(), ncomp_real * num_facets_global);
  ASSERT_EQ(test_global_int.size(), ncomp_int * num_facets_global);

  for (int fx = 0; fx < num_facets_global; fx++) {
    for (int rx = 0; rx < ncomp_real; rx++) {
      ASSERT_EQ(ptr_real[fx * ncomp_real + rx],
                test_global_real.at(fx * ncomp_real + rx));
    }
    for (int ix = 0; ix < ncomp_int; ix++) {
      ASSERT_EQ(ptr_int[fx * ncomp_int + ix],
                test_global_int.at(fx * ncomp_int + ix));
    }
  }

  b2d->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary2D, collect_2d) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 32;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));

  PetscInterface::generic_distribute(&dm);
  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);

  const int num_facets_global = b2d->get_num_facets_global();
  REAL *ptr_real = b2d->get_facets_real();
  int *ptr_int = b2d->get_facets_int();

  auto mesh_hierarchy = mesh->get_mesh_hierarchy();
  std::deque<std::pair<INT, double>> cells;

  std::map<INT, std::set<int>> map_mh_index_to_index;
  for (int ex = 0; ex < num_facets_global; ex++) {
    cells.clear();
    auto bb = b2d->wrap_get_bounding_box(ex);
    ExternalCommon::bounding_box_map(bb, mesh_hierarchy, cells);
    for (auto &cx_w : cells) {
      map_mh_index_to_index[cx_w.first].insert(ex);
    }
  }

  auto &required_mh_cells = b2d->get_required_mh_cells();
  auto &collected_mh_cells = b2d->get_collected_mh_cells();

  const INT mh_cell = map_mh_index_to_index.begin()->first;

  ASSERT_EQ(required_mh_cells.size(), 0);
  ASSERT_EQ(collected_mh_cells.size(), 0);

  required_mh_cells.insert(mh_cell);
  b2d->wrap_collect_cells();
  ASSERT_EQ(required_mh_cells.size(), 1);
  ASSERT_EQ(collected_mh_cells.size(), 1);

  auto d_map_edge_discovery = b2d->get_d_map_edge_discovery();
  auto d_map_edge_normals = b2d->get_d_map_edge_normals();

  TestBoundaryInteraction2D::BoundaryInteractionCellData2D cell_data;
  TestBoundaryInteraction2D::BoundaryInteractionNormalData2D normal_data;

  ASSERT_TRUE(d_map_edge_discovery->host_get(mh_cell, &cell_data));

  ASSERT_EQ(cell_data.num_edges, map_mh_index_to_index.at(mh_cell).size());

  const int num_edges = cell_data.num_edges;
  const int ncomp_real = b2d->get_ncomp_real();
  const int ncomp_int = b2d->get_ncomp_int();
  std::vector<REAL> h_real(num_edges * 4);
  std::vector<int> h_int(num_edges * ncomp_int);

  sycl_target->queue
      .memcpy(h_real.data(), cell_data.d_real, num_edges * 4 * sizeof(REAL))
      .wait_and_throw();
  sycl_target->queue
      .memcpy(h_int.data(), cell_data.d_int,
              num_edges * ncomp_int * sizeof(int))
      .wait_and_throw();

  auto lambda_find_int = [&](const int value) -> int {
    for (int ix = 0; ix < num_facets_global; ix++) {
      if (ptr_int[ix * ncomp_int + 1] == value) {
        return ix;
      }
    }
    return -1;
  };

  for (int edgex = 0; edgex < num_edges; edgex++) {
    const int group_id = h_int.at(edgex * ncomp_int + 0);
    const int facet_global_id = h_int.at(edgex * ncomp_int + 1);
    auto index = lambda_find_int(facet_global_id);
    ASSERT_TRUE(index > -1);
    ASSERT_EQ(ptr_real[index * ncomp_real + 0], h_real.at(edgex * 4 + 0));
    ASSERT_EQ(ptr_real[index * ncomp_real + 1], h_real.at(edgex * 4 + 1));
    ASSERT_EQ(ptr_real[index * ncomp_real + 2], h_real.at(edgex * 4 + 2));
    ASSERT_EQ(ptr_real[index * ncomp_real + 3], h_real.at(edgex * 4 + 3));
    ASSERT_EQ(1, h_int.at(edgex * ncomp_int + 0));
    ASSERT_EQ(ptr_int[index * ncomp_int + 1], h_int.at(edgex * ncomp_int + 1));

    ASSERT_TRUE(d_map_edge_normals->host_get(facet_global_id, &normal_data));
    REAL h_normal[2];
    sycl_target->queue.memcpy(h_normal, normal_data.d_normal, 2 * sizeof(REAL))
        .wait_and_throw();

    ASSERT_EQ(ptr_real[index * ncomp_real + 4], h_normal[0]);
    ASSERT_EQ(ptr_real[index * ncomp_real + 5], h_normal[1]);
  }

  b2d->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary2D, pre_integrate) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 8;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
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

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
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
      if (sx.first != correct_group) {
        ASSERT_EQ(sx.second->get_npart_local(), 0);
      }
    }

    // Check the boundary intersection is at the expected place
    particle_loop(
        sub_groups.at(correct_group),
        [=](auto INDEX, auto CURR_POSITION, auto PREV_POSITION,
            auto BOUNDARY_POSITION) {
          NESO_KERNEL_ASSERT(
              KERNEL_ABS(BOUNDARY_POSITION.at(component) - value) < 1.0e-14,
              k_ep);
        },
        Access::read(ParticleLoopIndex{}), Access::read(Sym<REAL>("P")),
        Access::read(b2d->previous_position_sym),
        Access::read(b2d->boundary_position_sym))
        ->execute();
    ASSERT_EQ(ep->get_flag(), 0);
    // Check the boundary metadata is as expected
    particle_loop(
        sub_groups.at(correct_group),
        [=](auto BOUNDARY_INFO) {
          NESO_KERNEL_ASSERT(BOUNDARY_INFO.at(0) > -1, k_ep);
          NESO_KERNEL_ASSERT(BOUNDARY_INFO.at(1) == correct_group, k_ep);
          NESO_KERNEL_ASSERT(BOUNDARY_INFO.at(2) > -1, k_ep);
        },
        Access::read(b2d->boundary_label_sym))
        ->execute();
    ASSERT_EQ(ep->get_flag(), 0);
    // Check that the normal for the intersection point is as expected
    auto normal_mapper = b2d->get_device_normal_mapper();
    particle_loop(
        sub_groups.at(correct_group),
        [=](auto B_C) {
          REAL *normal;
          NESO_KERNEL_ASSERT(normal_mapper.get(B_C.at(2), &normal), k_ep);

          const bool bx0 = KERNEL_ABS(normal[0] - normalx) < 1.0e-15;
          const bool bx1 = KERNEL_ABS(normal[0] + normalx) < 1.0e-15;
          const bool by0 = KERNEL_ABS(normal[1] - normaly) < 1.0e-15;
          const bool by1 = KERNEL_ABS(normal[1] + normaly) < 1.0e-15;

          NESO_KERNEL_ASSERT(bx0 || bx1, k_ep);
          NESO_KERNEL_ASSERT(by0 || by1, k_ep);
        },
        Access::read(b2d->boundary_label_sym))
        ->execute();
    ASSERT_EQ(ep->get_flag(), 0);
  }

  b2d->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary2D, corners) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int ndim = 2;
  const int mesh_size = 8;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
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
          NESO_KERNEL_ASSERT(KERNEL_ABS(B_P.at(0) - intersectx) < 1.0e-15,
                             k_ep);
          NESO_KERNEL_ASSERT(KERNEL_ABS(B_P.at(1) - intersecty) < 1.0e-15,
                             k_ep);
          NESO_KERNEL_ASSERT(B_C.at(0) > -1, k_ep);
          NESO_KERNEL_ASSERT(B_C.at(1) == 1, k_ep);
        },
        Access::read(b2d->boundary_position_sym),
        Access::read(b2d->boundary_label_sym))
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

TEST(PETScBoundary2D, reflection_truncated) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int nsteps = 200;
  const REAL dt = 0.1;

  const int ndim = 2;
  const int mesh_size = 8;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
                               lower, upper,
                               /* periodicity */ NULL, PETSC_TRUE, &dm));
  PetscInterface::generic_distribute(&dm);
  auto A = particle_loop_common(dm, 1024);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[0] = {1};
  boundary_groups[1] = {2};
  boundary_groups[2] = {3};
  boundary_groups[3] = {4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);
  auto reflection =
      std::make_shared<PetscInterface::BoundaryReflection>(b2d, 1.0e-10);

  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    auto sub_groups = b2d->post_integration(aa);

    for (auto &gx : sub_groups) {
      particle_loop(
          gx.second,
          [=](auto V, auto U) {
            U.at(0) = V.at(0);
            U.at(1) = V.at(1);
            U.at(2) = V.at(2);
          },
          Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("U")))
          ->execute();

      reflection->execute(gx.second, Sym<REAL>("P"), Sym<REAL>("V"),
                          Sym<REAL>("TSP"));

      particle_loop(
          gx.second,
          [=](auto V, auto U) {
            const REAL V_mag =
                V.at(0) * V.at(0) + V.at(1) * V.at(1) + V.at(2) * V.at(2);
            const REAL U_mag =
                U.at(0) * U.at(0) + U.at(1) * U.at(1) + U.at(2) * U.at(2);

            NESO_KERNEL_ASSERT(Kernel::abs(V_mag - U_mag) < 1.0e-14, k_ep);

            if ((Kernel::abs(V.at(0)) > 1.0e-14) &&
                (Kernel::abs(V.at(1)) > 1.0e-14)) {
              const bool x_flipped = Kernel::abs(V.at(0) + U.at(0)) < 1.0e-15;
              const bool y_flipped = Kernel::abs(V.at(1) + U.at(1)) < 1.0e-15;
              NESO_KERNEL_ASSERT(x_flipped != y_flipped, k_ep);
              if (x_flipped) {
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(1) - U.at(1)) < 1.0e-15,
                                   k_ep);
              }
              if (y_flipped) {
                NESO_KERNEL_ASSERT(Kernel::abs(V.at(0) - U.at(0)) < 1.0e-15,
                                   k_ep);
              }
            }
          },
          Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("U")))
          ->execute();
      ASSERT_EQ(ep->get_flag(), 0);

      particle_loop(
          gx.second,
          [=](auto P) {
            NESO_KERNEL_ASSERT((0.0 < P.at(0)) && (P.at(0) < mesh_size * h),
                               k_ep);
            NESO_KERNEL_ASSERT((0.0 < P.at(1)) && (P.at(1) < mesh_size * h),
                               k_ep);
          },
          Access::read(Sym<REAL>("P")))
          ->execute();
      ASSERT_EQ(ep->get_flag(), 0);
    }
  };

  auto lambda_apply_timestep_reset = [&](auto aa) {
    particle_loop(
        aa,
        [=](auto TSP) {
          TSP.at(0) = 0.0;
          TSP.at(1) = 0.0;
        },
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_apply_advection_step =
      [=](ParticleSubGroupSharedPtr iteration_set) -> void {
    particle_loop(
        "euler_advection", iteration_set,
        [=](auto V, auto P, auto TSP) {
          const REAL dt_left = dt - TSP.at(0);
          if (dt_left > 0.0) {
            P.at(0) += dt_left * V.at(0);
            P.at(1) += dt_left * V.at(1);
            TSP.at(0) = dt;
            TSP.at(1) = dt_left;
          }
        },
        Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")),
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_pre_advection = [&](auto aa) { b2d->pre_integration(aa); };
  auto lambda_find_partial_moves = [&](auto aa) {
    return static_particle_sub_group(
        A, [=](auto TSP) { return TSP.at(0) < dt; },
        Access::read(Sym<REAL>("TSP")));
  };
  auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
    const int size = aa->get_npart_local();
    return size > 0;
  };
  auto lambda_apply_timestep = [&](auto aa) {
    lambda_apply_timestep_reset(aa);
    lambda_pre_advection(aa);
    lambda_apply_advection_step(aa);
    lambda_apply_boundary_conditions(aa);
    aa = lambda_find_partial_moves(aa);
    while (lambda_partial_moves_remaining(aa)) {
      lambda_pre_advection(aa);
      lambda_apply_advection_step(aa);
      lambda_apply_boundary_conditions(aa);
      aa = lambda_find_partial_moves(aa);
    }
  };

  for (int stepx = 0; stepx < nsteps; stepx++) {
    lambda_apply_timestep(static_particle_sub_group(A));
    A->hybrid_move();
    A->cell_move();
  }

  b2d->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETScBoundary2D, reflection_advection) {

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;

  const int nsteps = 500;
  const REAL dt = 0.01;

  const int ndim = 2;
  const int mesh_size = 32;
  const REAL h = 1.0;
  PetscInt faces[3] = {mesh_size, mesh_size, mesh_size};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {mesh_size * h, mesh_size * h, mesh_size * h};

  // PETSCCHK(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces,
  //                              lower, upper,
  //                              /* periodicity */ NULL, PETSC_TRUE, &dm));

  PETSCCHK(
      DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                               "/home/js0259/git-ukaea/NESO-Particles-paper/"
                               "resources/mesh_ring/mesh_ring.msh",
                               (PetscBool)1, &dm));

  PetscInterface::generic_distribute(&dm);
  auto A = particle_loop_common(dm, 2048);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  auto b2d = std::make_shared<TestBoundaryInteraction2D>(sycl_target, mesh,
                                                         boundary_groups);
  auto reflection =
      std::make_shared<PetscInterface::BoundaryReflection>(b2d, 1.0e-10);

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    auto sub_groups = b2d->post_integration(aa);
    for (auto &gx : sub_groups) {
      reflection->execute(gx.second, Sym<REAL>("P"), Sym<REAL>("V"),
                          Sym<REAL>("TSP"));
    }
  };
  auto lambda_apply_timestep_reset = [&](auto aa) {
    particle_loop(
        aa,
        [=](auto TSP) {
          TSP.at(0) = 0.0;
          TSP.at(1) = 0.0;
        },
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_apply_advection_step =
      [=](ParticleSubGroupSharedPtr iteration_set) -> void {
    particle_loop(
        "euler_advection", iteration_set,
        [=](auto V, auto P, auto TSP) {
          const REAL dt_left = dt - TSP.at(0);
          if (dt_left > 0.0) {
            P.at(0) += dt_left * V.at(0);
            P.at(1) += dt_left * V.at(1);
            TSP.at(0) = dt;
            TSP.at(1) = dt_left;
          }
        },
        Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("P")),
        Access::write(Sym<REAL>("TSP")))
        ->execute();
  };
  auto lambda_pre_advection = [&](auto aa) { b2d->pre_integration(aa); };
  auto lambda_find_partial_moves = [&](auto aa) {
    return static_particle_sub_group(
        A, [=](auto TSP) { return TSP.at(0) < dt; },
        Access::read(Sym<REAL>("TSP")));
  };
  auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
    const int size = aa->get_npart_local();
    return size > 0;
  };
  auto lambda_apply_timestep = [&](auto aa) {
    lambda_apply_timestep_reset(aa);
    lambda_pre_advection(aa);
    lambda_apply_advection_step(aa);
    lambda_apply_boundary_conditions(aa);
    aa = lambda_find_partial_moves(aa);
    while (lambda_partial_moves_remaining(aa)) {
      lambda_pre_advection(aa);
      lambda_apply_advection_step(aa);
      lambda_apply_boundary_conditions(aa);
      aa = lambda_find_partial_moves(aa);
    }
  };

  H5Part h5part("traj_reflection.h5part", A, Sym<REAL>("P"), Sym<REAL>("V"));

  for (int stepx = 0; stepx < nsteps; stepx++) {
    if ((stepx % 20 == 0) && (sycl_target->comm_pair.rank_parent == 0)) {
      nprint("step:", stepx);
    }
    lambda_apply_timestep(static_particle_sub_group(A));
    A->hybrid_move();
    A->cell_move();
    h5part.write();
  }
  h5part.close();

  b2d->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
