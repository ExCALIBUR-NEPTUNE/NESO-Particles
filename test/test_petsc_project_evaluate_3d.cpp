#ifdef NESO_PARTICLES_PETSC
#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

namespace {

ParticleGroupSharedPtr particle_loop_common(const int ndim, DM dm,
                                            const int N = 1093) {

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
                             ParticleProp(Sym<REAL>("MASS"), 1),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
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

void wrapper_mesh_project_3d_dg0(const int ndim, DM dm) {

  auto A = particle_loop_common(ndim, dm, 10930);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  auto dgpe = std::make_shared<PetscInterface::DMPlexProjectEvaluateDG>(
      mesh, sycl_target, "DG", 0);

  const int cell_count = mesh->get_cell_count();

  std::vector<REAL> h_dofs(cell_count * 3);
  std::vector<REAL> h_inverse_volumes(cell_count);
  for (int ix = 0; ix < cell_count; ix++) {
    for (int dx = 0; dx < 3; dx++) {
      h_dofs.at(ix * 3 + dx) = ix * 3 + dx;
    }
    h_inverse_volumes.at(ix) = 1.0 / mesh->dmh->get_cell_volume(ix);
  }

  dgpe->set_dofs(3, h_dofs);
  fill(A, Sym<REAL>("U"), 0.0);
  dgpe->evaluate(A, Sym<REAL>("U"));

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      A,
      [=](auto INDEX, auto U) {
        const int cell = INDEX.cell;
        for (int dx = 0; dx < 3; dx++) {
          NESO_KERNEL_ASSERT(Kernel::abs(U.at(dx) - (cell * 3 + dx)) < 1.0e-15,
                             k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<REAL>("U")))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  particle_loop(
      A,
      [=](auto INDEX, auto MASS) {
        MASS.at(0) = 1.0 + 0.001 * INDEX.cell +
                     0.1 * (INDEX.get_local_linear_index() % 31);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("MASS")))
      ->execute();

  auto d_inverse_volumes =
      std::make_shared<LocalArray<REAL>>(sycl_target, h_inverse_volumes);
  auto d_project_correct =
      std::make_shared<LocalArray<REAL>>(sycl_target, h_inverse_volumes);
  d_project_correct->fill(0.0);

  particle_loop(
      A,
      [=](auto INDEX, auto MASS, auto IV, auto PROJ) {
        const REAL contrib = MASS.at(0);
        const auto cell = INDEX.cell;
        const REAL inverse_volume = IV.at(cell);
        PROJ.fetch_add(cell, inverse_volume * contrib);
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<REAL>("MASS")),
      Access::read(d_inverse_volumes), Access::add(d_project_correct))
      ->execute();

  auto h_project_correct = d_project_correct->get();
  dgpe->project(A, Sym<REAL>("MASS"));

  std::vector<REAL> h_project_to_test;
  dgpe->get_dofs(1, h_project_to_test);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const REAL correct = h_project_correct.at(cellx);
    const REAL to_test = h_project_to_test.at(cellx);
    const REAL err = relative_error(correct, to_test);
    ASSERT_TRUE(err < 1.0e-12);
  }

  // auto vtk_data = dgpe->get_vtk_data("u");
  // VTK::VTKHDF v("foo.vtkhdf", mesh->get_comm());
  // v.write(vtk_data);
  // v.close();

  sycl_target->free();
  mesh->free();
}

} // namespace

TEST(PETSc, dmplex_project_evaluate_dg_3d_box) {
  const int ndim = 3;
  PETSCCHK(PetscInitializeNoArguments());
  const int mesh_size = 16;
  const REAL h = 1.41;
  PetscInt faces[3] = {mesh_size, mesh_size - 1, mesh_size - 2};
  PetscReal lower[3] = {0.0, 0.0, 0.0};
  PetscReal upper[3] = {faces[0] * h, faces[1] * h, faces[2] * h};
  DM dm;
  PETSCCHK(NPPETScAPI::NP_DMPlexCreateBoxMesh(
      PETSC_COMM_WORLD, ndim, PETSC_FALSE, faces, lower, upper,
      /* periodicity */ NULL, PETSC_TRUE, &dm));
  PetscInterface::generic_distribute(&dm);

  wrapper_mesh_project_3d_dg0(ndim, dm);
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

TEST(PETSc, dmplex_project_evaluate_dg_3d_ref_mesh) {
  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  wrapper_mesh_project_3d_dg0(3, dm);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
