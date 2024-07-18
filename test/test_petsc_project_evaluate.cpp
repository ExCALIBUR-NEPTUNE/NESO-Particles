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

inline REAL barycentric_test_function(const REAL x, const REAL y) {
  return 4.0 + 3.0 * x - 2.0 * y;
}

TEST(PETSC, dmplex_project_evaluate_barycentric) {
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

  // Add quadrature 3 points per cell
  PetscInterface::uniform_within_dmplex_cells(mesh, 3, positions, cells);
  qpm->add_points_initialise();
  const int npart_qpm = positions.at(0).size();
  REAL point[2];
  for (int px = 0; px < npart_qpm; px++) {
    point[0] = positions.at(0).at(px);
    point[1] = positions.at(1).at(px);
    qpm->add_point(point);
  }
  qpm->add_points_finalise();

  // move the points to the vertices
  auto cdc_vertices =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 4, 2);
  auto cdc_num_vertices =
      std::make_shared<CellDatConst<int>>(sycl_target, cell_count, 1, 1);
  std::vector<std::vector<REAL>> vertices;
  for (int cx = 0; cx < cell_count; cx++) {
    mesh->dmh->get_cell_vertices(cx, vertices);
    const int num_verts = vertices.size();
    cdc_num_vertices->set_value(cx, 0, 0, num_verts);
    for (int vx = 0; vx < 3; vx++) {
      for (int dx = 0; dx < 2; dx++) {
        cdc_vertices->set_value(cx, vx, dx, vertices.at(vx).at(dx));
      }
    }
  }
  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      qpm->particle_group,
      [=](auto INDEX, auto VERTICES, auto POS) {
        REAL l[3] = {0.0, 0.0, 0.0};
        NESO_KERNEL_ASSERT(INDEX.layer < 3, k_ep);
        l[INDEX.layer] = 1.0;
        REAL x, y;
        ExternalCommon::triangle_barycentric_to_cartesian(
            VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
            VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1), l[0], l[1],
            l[2], &x, &y);
        POS.at(0) = x;
        POS.at(1) = y;
      },
      Access::read(ParticleLoopIndex{}), Access::read(cdc_vertices),
      Access::write(qpm->particle_group->position_dat))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

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
  qpm->free();
  sycl_target->free();
  mesh->free();
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
