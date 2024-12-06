#ifdef NESO_PARTICLES_PETSC
#include "include/test_neso_particles.hpp"

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

inline REAL barycentric_test_function_linear0(const REAL x, const REAL y) {
  return 4.0 + 3.0 * x - 2.0 * y;
}
inline REAL barycentric_test_function_linear1(const REAL x, const REAL y) {
  return 40.0 + 30.0 * x - 20.0 * y;
}

inline REAL barycentric_test_function_bilinear0(const REAL x, const REAL y) {
  return 4.0 + 3.0 * x - 2.0 * y + 5.0 * x * y;
}

inline REAL barycentric_test_function_bilinear1(const REAL x, const REAL y) {
  return 40.0 + 30.0 * x - 20.0 * y + 50.0 * x * y;
}

TEST(PETSC, dmplex_project_evaluate_qpm_vertex) {

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
  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());
  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  auto cell_vertices_info = get_cell_vertices_cdc(sycl_target, mesh->dmh);
  auto cdc_num_vertices = std::get<0>(cell_vertices_info);
  auto cdc_vertices = std::get<1>(cell_vertices_info);

  auto qpm =
      PetscInterface::make_quadrature_point_mapper_vertex(sycl_target, domain);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

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

  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, dmplex_project_evaluate_qpm_average) {

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

  auto qpm =
      PetscInterface::make_quadrature_point_mapper_average(sycl_target, domain);

  std::vector<REAL> average(ndim);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto sym = qpm->particle_group->position_dat->sym;
    auto W = qpm->particle_group->get_cell(sym, cellx);
    mesh->dmh->get_cell_vertex_average(cellx, average);
    ASSERT_EQ(W->nrow, 1);
    for (int dx = 0; dx < ndim; dx++) {
      ASSERT_NEAR(W->at(0, dx), average.at(dx), 1.0e-15);
    }
  }

  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, dmplex_evaluate_barycentric) {

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

  auto qpm =
      PetscInterface::make_quadrature_point_mapper_vertex(sycl_target, domain);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true), ParticleProp(Sym<REAL>("Q"), 1),
      ParticleProp(Sym<REAL>("Q2"), 2), ParticleProp(Sym<REAL>("R"), 1),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
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

  auto lambda_l0 = [=](auto x, auto y) -> REAL {
    return barycentric_test_function_linear0(x, y);
  };
  auto lambda_l1 = [=](auto x, auto y) -> REAL {
    return barycentric_test_function_linear1(x, y);
  };
  auto lambda_b0 = [=](auto x, auto y) -> REAL {
    return barycentric_test_function_bilinear0(x, y);
  };
  auto lambda_b1 = [=](auto x, auto y) -> REAL {
    return barycentric_test_function_bilinear1(x, y);
  };

  // Linear test function over the domain should be exactly representable in
  // both quads and triangles.
  ExternalCommon::interpolate<2, 1>(qpm, lambda_l0);

  dpe->evaluate(A, Sym<REAL>("Q"));
  for (int cx = 0; cx < cell_count; cx++) {
    auto P = A->get_cell(Sym<REAL>("P"), cx);
    auto Q = A->get_cell(Sym<REAL>("Q"), cx);
    for (int rx = 0; rx < P->nrow; rx++) {
      const REAL correct =
          barycentric_test_function_linear0(P->at(rx, 0), P->at(rx, 1));
      const REAL to_test = Q->at(rx, 0);
      const REAL scaling = std::max(1.0, std::abs(correct));
      ASSERT_NEAR(correct / scaling, to_test / scaling, 1.0e-10);
    }
  }

  // Bilinear function exactly representable in the quads only.
  ExternalCommon::interpolate<2>(qpm, 1, 0, lambda_b0);

  dpe->evaluate(A, Sym<REAL>("Q"));
  for (int cx = 0; cx < cell_count; cx++) {
    const auto num_vertices = cdc_num_vertices->get_value(cx, 0, 0);
    if (num_vertices == 4) {
      auto P = A->get_cell(Sym<REAL>("P"), cx);
      auto Q = A->get_cell(Sym<REAL>("Q"), cx);
      for (int rx = 0; rx < P->nrow; rx++) {
        const REAL correct =
            barycentric_test_function_bilinear0(P->at(rx, 0), P->at(rx, 1));
        const REAL to_test = Q->at(rx, 0);
        const REAL scaling = std::max(1.0, std::abs(correct));
        ASSERT_NEAR(correct / scaling, to_test / scaling, 1.0e-10);
      }
    }
  }

  // 2 components
  // linear
  ExternalCommon::interpolate<2, 2>(qpm, lambda_l0, lambda_l1);

  dpe->evaluate(A, Sym<REAL>("Q2"));
  for (int cx = 0; cx < cell_count; cx++) {
    auto P = A->get_cell(Sym<REAL>("P"), cx);
    auto Q2 = A->get_cell(Sym<REAL>("Q2"), cx);
    for (int rx = 0; rx < P->nrow; rx++) {
      const REAL correct0 =
          barycentric_test_function_linear0(P->at(rx, 0), P->at(rx, 1));
      const REAL correct1 =
          barycentric_test_function_linear1(P->at(rx, 0), P->at(rx, 1));
      {
        const REAL to_test = Q2->at(rx, 0);
        const REAL scaling = std::max(1.0, std::abs(correct0));
        ASSERT_NEAR(correct0 / scaling, to_test / scaling, 1.0e-10);
      }
      {
        const REAL to_test = Q2->at(rx, 1);
        const REAL scaling = std::max(1.0, std::abs(correct1));
        ASSERT_NEAR(correct1 / scaling, to_test / scaling, 1.0e-10);
      }
    }
  }

  // bilinear
  ExternalCommon::interpolate<2>(qpm, 2, 0, lambda_b0);
  ExternalCommon::interpolate<2>(qpm, 2, 1, lambda_b1);

  dpe->evaluate(A, Sym<REAL>("Q2"));
  for (int cx = 0; cx < cell_count; cx++) {
    const auto num_vertices = cdc_num_vertices->get_value(cx, 0, 0);
    if (num_vertices == 4) {
      auto P = A->get_cell(Sym<REAL>("P"), cx);
      auto Q2 = A->get_cell(Sym<REAL>("Q2"), cx);
      for (int rx = 0; rx < P->nrow; rx++) {
        {
          const REAL correct =
              barycentric_test_function_bilinear0(P->at(rx, 0), P->at(rx, 1));
          const REAL to_test = Q2->at(rx, 0);
          const REAL scaling = std::max(1.0, std::abs(correct));
          ASSERT_NEAR(correct / scaling, to_test / scaling, 1.0e-10);
        }
        {
          const REAL correct =
              barycentric_test_function_bilinear1(P->at(rx, 0), P->at(rx, 1));
          const REAL to_test = Q2->at(rx, 1);
          const REAL scaling = std::max(1.0, std::abs(correct));
          ASSERT_NEAR(correct / scaling, to_test / scaling, 1.0e-10);
        }
      }
    }
  }

  A->free();
  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, dmplex_project_barycentric_dg) {
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

  auto qpm =
      PetscInterface::make_quadrature_point_mapper_vertex(sycl_target, domain);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true), ParticleProp(Sym<REAL>("Q"), 1),
      ParticleProp(Sym<REAL>("Q2"), 2), ParticleProp(Sym<REAL>("R"), 1),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  const int npart_per_cell = 17;
  PetscInterface::uniform_within_dmplex_cells(mesh, npart_per_cell, positions,
                                              cells);
  const int N = cells.size();
  ParticleSet initial_distribution(N, particle_spec);

  std::mt19937 rng_state(52234234 + sycl_target->comm_pair.rank_parent);
  std::normal_distribution<REAL> rng_dist(0, 1.0);
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("Q")][px][0] = rng_dist(rng_state);
    initial_distribution[Sym<REAL>("Q2")][px][0] = rng_dist(rng_state);
    initial_distribution[Sym<REAL>("Q2")][px][1] = 20.0 + rng_dist(rng_state);
  }
  A->add_particles_local(initial_distribution);

  auto dpe = std::make_shared<PetscInterface::DMPlexProjectEvaluateBarycentric>(
      qpm, "Barycentric", 1, true);
  auto dpe_dg =
      std::make_shared<PetscInterface::DMPlexProjectEvaluate>(qpm, "DG", 0);

  // Test that the integral over each cell matches the DG0 projection
  dpe->project(A, Sym<REAL>("Q2"));

  // Average value of the vertex coeffs
  std::vector<REAL> to_test0(cell_count);
  std::vector<REAL> to_test1(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    REAL v0 = 0.0;
    REAL v1 = 0.0;
    auto Q2 = qpm->particle_group->get_cell(qpm->get_sym(2), cx);
    const auto num_vertices = cdc_num_vertices->get_value(cx, 0, 0);
    for (int rx = 0; rx < Q2->nrow; rx++) {
      v0 += Q2->at(rx, 0) / num_vertices;
      v1 += Q2->at(rx, 1) / num_vertices;
    }
    to_test0.at(cx) = v0;
    to_test1.at(cx) = v1;
  }
  dpe_dg->project(A, Sym<REAL>("Q2"));
  std::vector<REAL> correct0(cell_count);
  std::vector<REAL> correct1(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    REAL v0 = 0.0;
    REAL v1 = 0.0;
    auto Q2 = qpm->particle_group->get_cell(qpm->get_sym(2), cx);
    const auto num_vertices = cdc_num_vertices->get_value(cx, 0, 0);
    for (int rx = 0; rx < Q2->nrow; rx++) {
      v0 += Q2->at(rx, 0) / num_vertices;
      v1 += Q2->at(rx, 1) / num_vertices;
    }
    correct0.at(cx) = v0;
    correct1.at(cx) = v1;
  }
  for (int cx = 0; cx < cell_count; cx++) {
    {
      const REAL scaling = std::max(1.0, std::abs(correct0.at(cx)));
      ASSERT_NEAR(to_test0.at(cx) / scaling, correct0.at(cx) / scaling,
                  1.0e-10);
    }
    {
      const REAL scaling = std::max(1.0, std::abs(correct1.at(cx)));
      ASSERT_NEAR(to_test1.at(cx) / scaling, correct1.at(cx) / scaling,
                  1.0e-10);
    }
  }

  A->free();
  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSC, dmplex_project_barycentric_coeffs) {

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

  auto qpm =
      PetscInterface::make_quadrature_point_mapper_vertex(sycl_target, domain);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true), ParticleProp(Sym<REAL>("Q"), 1),
      ParticleProp(Sym<REAL>("Q2"), 2), ParticleProp(Sym<REAL>("R"), 1),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  const int npart_per_cell = 17;
  PetscInterface::uniform_within_dmplex_cells(mesh, npart_per_cell, positions,
                                              cells);
  const int N = cells.size();
  ParticleSet initial_distribution(N, particle_spec);

  std::mt19937 rng_state(52234234 + sycl_target->comm_pair.rank_parent);
  std::normal_distribution<REAL> rng_dist(0, 1.0);
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<REAL>("Q")][px][0] = rng_dist(rng_state);
    initial_distribution[Sym<REAL>("Q2")][px][0] = rng_dist(rng_state);
    initial_distribution[Sym<REAL>("Q2")][px][1] = 20.0 + rng_dist(rng_state);
  }
  A->add_particles_local(initial_distribution);

  auto dpe = std::make_shared<PetscInterface::DMPlexProjectEvaluateBarycentric>(
      qpm, "Barycentric", 1, true);
  dpe->project(A, Sym<REAL>("Q2"));

  auto lambda_weights = [&](const int num_vertices, auto vertices, const REAL x,
                            const REAL y, REAL *l0, REAL *l1, REAL *l2,
                            REAL *l3) {
    if (num_vertices == 4) {
      REAL eta0, eta1;
      ExternalCommon::quad_cartesian_to_collapsed(
          vertices->at(0, 0), vertices->at(0, 1), vertices->at(1, 0),
          vertices->at(1, 1), vertices->at(2, 0), vertices->at(2, 1),
          vertices->at(3, 0), vertices->at(3, 1), x, y, &eta0, &eta1);
      REAL xx, yy;
      ExternalCommon::quad_collapsed_to_cartesian(
          vertices->at(0, 0), vertices->at(0, 1), vertices->at(1, 0),
          vertices->at(1, 1), vertices->at(2, 0), vertices->at(2, 1),
          vertices->at(3, 0), vertices->at(3, 1), eta0, eta1, &xx, &yy);
      ASSERT_NEAR(x, xx, 1.0e-14);
      ASSERT_NEAR(y, yy, 1.0e-14);

      ExternalCommon::quad_cartesian_to_barycentric(
          vertices->at(0, 0), vertices->at(0, 1), vertices->at(1, 0),
          vertices->at(1, 1), vertices->at(2, 0), vertices->at(2, 1),
          vertices->at(3, 0), vertices->at(3, 1), x, y, l0, l1, l2, l3);

      REAL lx = (*l0) * vertices->at(0, 0) + (*l1) * vertices->at(1, 0) +
                (*l2) * vertices->at(2, 0) + (*l3) * vertices->at(3, 0);
      REAL ly = (*l0) * vertices->at(0, 1) + (*l1) * vertices->at(1, 1) +
                (*l2) * vertices->at(2, 1) + (*l3) * vertices->at(3, 1);

      ASSERT_NEAR(x, lx, 1.0e-14);
      ASSERT_NEAR(y, ly, 1.0e-14);

    } else {

      ExternalCommon::triangle_cartesian_to_barycentric(
          vertices->at(0, 0), vertices->at(0, 1), vertices->at(1, 0),
          vertices->at(1, 1), vertices->at(2, 0), vertices->at(2, 1), x, y, l0,
          l1, l2);

      REAL xx, yy;

      ExternalCommon::triangle_barycentric_to_cartesian(
          vertices->at(0, 0), vertices->at(0, 1), vertices->at(1, 0),
          vertices->at(1, 1), vertices->at(2, 0), vertices->at(2, 1), *l0, *l1,
          *l2, &xx, &yy);

      ASSERT_NEAR(x, xx, 1.0e-14);
      ASSERT_NEAR(y, yy, 1.0e-14);
    }
  };

  // recompute the vertex coeffs and check they match
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto vertices = cdc_vertices->get_cell(cellx);
    auto num_vertices = cdc_num_vertices->get_value(cellx, 0, 0);
    auto P = A->get_cell(Sym<REAL>("P"), cellx);
    auto Q2 = A->get_cell(Sym<REAL>("Q2"), cellx);
    const int num_particles = P->nrow;
    std::vector<std::vector<REAL>> coeffs = {{0.0, 0.0, 0.0, 0.0},
                                             {0.0, 0.0, 0.0, 0.0}};

    for (int px = 0; px < num_particles; px++) {
      REAL l[4];
      const REAL x = P->at(px, 0);
      const REAL y = P->at(px, 1);
      lambda_weights(num_vertices, vertices, x, y, &l[0], &l[1], &l[2], &l[3]);
      for (int cx = 0; cx < 2; cx++) {
        for (int vx = 0; vx < num_vertices; vx++) {
          coeffs.at(cx).at(vx) += l[vx] * Q2->at(px, cx);
        }
      }
    }

    const REAL volume = mesh->dmh->get_cell_volume(cellx);
    for (int cx = 0; cx < 2; cx++) {
      for (int vx = 0; vx < num_vertices; vx++) {
        coeffs.at(cx).at(vx) *= (num_vertices / volume);
      }
    }

    auto Q2project = qpm->particle_group->get_cell(qpm->get_sym(2), cellx);
    for (int cx = 0; cx < 2; cx++) {
      for (int vx = 0; vx < num_vertices; vx++) {
        const auto to_test = Q2project->at(vx, cx);
        const auto correct = coeffs.at(cx).at(vx);
        const REAL scaling =
            std::abs(correct) > 0.0 ? 1.0 / std::abs(correct) : 1.0;
        const REAL err = std::abs(to_test - correct) * scaling;
        ASSERT_TRUE(err < 1.0e-10);
      }
    }
  }

  A->free();
  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
