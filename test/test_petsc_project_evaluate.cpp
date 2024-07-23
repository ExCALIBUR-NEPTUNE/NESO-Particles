#ifdef NESO_PARTICLES_PETSC
#include "include/test_neso_particles.hpp"

/* TODO
TEST(PETSC, dmplex_project_evaluate_dg) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/reference_all_types_square_0.2.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  const int ndim = mesh->get_ndim();
  const int cell_count = mesh->get_cell_count();
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  auto qpm = std::make_shared<ExternalCommon::QuadraturePointMapper>(
      sycl_target, domain);
  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  PetscInterface::uniform_within_dmplex_cells(mesh, 2, positions, cells);

  qpm->add_points_initialise();
  const int npart_qpm = positions.at(0).size();
  REAL point[2];
  for (int px = 0; px < npart_qpm; px++) {
    point[0] = positions.at(0).at(px);
    point[1] = positions.at(1).at(px);
    qpm->add_point(point);
  }
  qpm->add_points_finalise();

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("R"), 1),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int npart_per_cell = 17;
  PetscInterface::uniform_within_dmplex_cells(mesh, npart_per_cell, positions,
                                              cells);
  const int N = cells.size();
  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("Q")][px][0] = cells.at(px) + 1.0;
  }
  A->add_particles_local(initial_distribution);

  auto dpe =
      std::make_shared<PetscInterface::DMPlexProjectEvaluate>(qpm, "DG", 0);
  dpe->project(A, Sym<REAL>("Q"));

  for (int cx = 0; cx < cell_count; cx++) {
    auto Q = qpm->particle_group->get_cell(qpm->get_sym(1), cx);
    for (int rx = 0; rx < Q->nrow; rx++) {
      ASSERT_NEAR(Q->at(rx, 0),
                  (cx + 1) * npart_per_cell / mesh->dmh->get_cell_volume(cx),
                  1.0e-10);
    }
  }

  // scale the quadrature point values
  const REAL k_scale = 3.14;
  particle_loop(
      qpm->particle_group, [=](auto Q) { Q.at(0) *= k_scale; },
      Access::write(Sym<REAL>(qpm->get_sym(1))))
      ->execute();

  // evalutate the scale quadrature point values
  dpe->evaluate(A, Sym<REAL>("R"));

  for (int cx = 0; cx < cell_count; cx++) {
    auto R = A->get_cell(Sym<REAL>("R"), cx);
    for (int rx = 0; rx < R->nrow; rx++) {
      const REAL correct =
          k_scale * (cx + 1) * npart_per_cell / mesh->dmh->get_cell_volume(cx);
      const REAL rescale = 1.0 / std::max(1.0, std::abs(correct));
      ASSERT_NEAR(R->at(rx, 0) * rescale, correct * rescale, 1.0e-10);
    }
  }

  A->free();
  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

*/

inline REAL barycentric_test_function(const REAL x, const REAL y) {
  return 4.0 + 3.0 * x - 2.0 * y;
}

TEST(PETSC, dmplex_project_evaluate_barycentric) {
  
  nprint("TODO UNCOMMENT DG TEST");

  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/reference_all_types_square_0.2.msh");

  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  auto mesh =
      std::make_shared<PetscInterface::DMPlexInterface>(dm, 0, MPI_COMM_WORLD);
  const int ndim = mesh->get_ndim();
  const int cell_count = mesh->get_cell_count();
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  auto cell_vertices_info = get_cell_vertices_cdc(sycl_target, mesh->dmh);
  auto cdc_num_vertices = std::get<0>(cell_vertices_info);
  auto cdc_vertices = std::get<1>(cell_vertices_info);

  auto qpm = std::make_shared<ExternalCommon::QuadraturePointMapper>(
      sycl_target, domain);
  std::vector<std::vector<double>> positions;
  std::vector<int> cells;

  // Add a point per vertex in the middle of each cell
  qpm->add_points_initialise();
  REAL point[2];
  std::vector<REAL> average(2);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int num_vertices = cdc_num_vertices->get_value(cellx, 0, 0);
    mesh->dmh->get_cell_vertex_average(cellx, average);
    for (int vx = 0; vx < num_vertices; vx++) {
      qpm->add_point(average.data());
    }
  }
  qpm->add_points_finalise();

  // move the points to the vertices
  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      qpm->particle_group,
      [=](auto INDEX, auto NUM_VERTICES, auto VERTICES, auto POS) {
        const auto num_vertices = NUM_VERTICES.at(0, 0);
        NESO_KERNEL_ASSERT(INDEX.layer < num_vertices, k_ep);
        POS.at(0) = VERTICES.at(INDEX.layer, 0);
        POS.at(1) = VERTICES.at(INDEX.layer, 1);
      },
      Access::read(ParticleLoopIndex{}), Access::read(cdc_num_vertices),
      Access::read(cdc_vertices),
      Access::write(qpm->particle_group->position_dat))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  // check bary coords
  particle_loop(
      qpm->particle_group,
      [=](auto INDEX, auto NUM_VERTICES, auto VERTICES, auto POS) {
        const auto num_vertices = NUM_VERTICES.at(0, 0);
        REAL l[4] = {0, 0, 0, 0};

        if (num_vertices == 4) {
          ExternalCommon::quad_cartesian_to_barycentric(
              VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
              VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1),
              VERTICES.at(3, 0), VERTICES.at(3, 1), POS.at(0), POS.at(1), &l[0],
              &l[1], &l[2], &l[3]);
        } else {
          ExternalCommon::triangle_cartesian_to_barycentric(
              VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
              VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1),
              POS.at(0), POS.at(1), &l[0], &l[1], &l[2]);
        }

        int index = -1;
        for (int vx = 0; vx < num_vertices; vx++) {
          if (Kernel::abs(l[vx] - 1.0) < 1.0e-6) {
            index = vx;
          }
        }
        NESO_KERNEL_ASSERT(index > -1, k_ep);
        for (int vx = 0; vx < num_vertices; vx++) {
          if (vx == index) {
            const bool cond = Kernel::abs(l[vx] - 1.0) < 1.0e-12;
            NESO_KERNEL_ASSERT(cond, k_ep);
          } else {
            const bool cond = Kernel::abs(l[vx]) < 1.0e-12;
            NESO_KERNEL_ASSERT(cond, k_ep);
            NESO_KERNEL_ASSERT(index == INDEX.layer, k_ep);
          }
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(cdc_num_vertices),
      Access::read(cdc_vertices),
      Access::write(qpm->particle_group->position_dat))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  /*



  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("Q"), 1),
                             ParticleProp(Sym<REAL>("R"), 1),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int npart_per_cell = 17;
  PetscInterface::uniform_within_dmplex_cells(mesh, npart_per_cell, positions,
                                              cells);
  const int N = cells.size();
  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("Q")][px][0] = cells.at(px) + 1.0;
  }
  A->add_particles_local(initial_distribution);

  auto dpe = std::make_shared<PetscInterface::DMPlexProjectEvaluateBarycentric>(
      qpm, "Barycentric", 1, true);
  auto dpe_dg =
      std::make_shared<PetscInterface::DMPlexProjectEvaluate>(qpm, "DG", 0);

  // Test that the integral over each cell matches the DG0 projection
  dpe->project(A, Sym<REAL>("Q"));
  std::vector<REAL> to_test(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    REAL v = 0.0;
    auto Q = qpm->particle_group->get_cell(qpm->get_sym(1), cx);
    if (cdc_num_vertices->get_value(cx, 0, 0) == 3) {
      ASSERT_EQ(Q->nrow, 3);
      for (int rx = 0; rx < Q->nrow; rx++) {
        v += Q->at(rx, 0) / 3.0;
      }
      to_test.at(cx) = v;
    }
  }
  dpe_dg->project(A, Sym<REAL>("Q"));
  std::vector<REAL> correct(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    REAL v = 0.0;
    auto Q = qpm->particle_group->get_cell(qpm->get_sym(1), cx);
    for (int rx = 0; rx < Q->nrow; rx++) {
      v += Q->at(rx, 0) / 3.0;
    }
    correct.at(cx) = v;
  }
  for (int cx = 0; cx < cell_count; cx++) {
    if (cdc_num_vertices->get_value(cx, 0, 0) == 3) {
      const REAL scaling = std::max(1.0, std::abs(correct.at(cx)));
      ASSERT_NEAR(to_test.at(cx) / scaling, correct.at(cx) / scaling, 1.0e-10);
    }
  }

  // Linear test function over the domain should be exactly representable.
  particle_loop(
      qpm->particle_group,
      [=](auto POS, auto Q) {
        Q.at(0) = barycentric_test_function(POS.at(0), POS.at(1));
      },
      Access::read(qpm->particle_group->position_dat),
      Access::write(qpm->get_sym(1)))
      ->execute();

  dpe->evaluate(A, Sym<REAL>("Q"));
  for (int cx = 0; cx < cell_count; cx++) {
    if (cdc_num_vertices->get_value(cx, 0, 0) == 3) {
      auto P = A->get_cell(Sym<REAL>("P"), cx);
      auto Q = A->get_cell(Sym<REAL>("Q"), cx);
      for (int rx = 0; rx < P->nrow; rx++) {
        const REAL correct =
            barycentric_test_function(P->at(rx, 0), P->at(rx, 1));
        const REAL to_test = Q->at(rx, 0);
        const REAL scaling = std::max(1.0, std::abs(correct));
        ASSERT_NEAR(correct / scaling, to_test / scaling, 1.0e-10);
      }
    }
  }

  A->free();
  */
  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
