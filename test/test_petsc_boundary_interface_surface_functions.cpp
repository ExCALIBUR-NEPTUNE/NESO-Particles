#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

TEST(PETScBoundary2D, setup_surface_functions) {

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

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2};
  boundary_groups[2] = {3, 4};

  auto b2d = std::make_shared<PetscInterface::BoundaryInteraction2D>(
      sycl_target, mesh, boundary_groups);

  auto f1 = b2d->create_function(1, "DG", 0);
  auto f2 = b2d->create_function(2, "DG", 0);

  std::set<int> s1 = {1, 2};
  std::set<int> s2 = {3, 4};
  auto face_sets = mesh->dmh->get_face_sets();

  std::set<PetscInt> cg1;
  std::set<PetscInt> cg2;

  for (auto labelx_indices : face_sets) {
    const auto label = labelx_indices.first;
    const auto &indices = labelx_indices.second;
    if (label > 0) {
      for (auto index : indices) {
        if (s1.count(label)) {
          cg1.insert(mesh->dmh->get_point_global_index(index));
        } else {
          ASSERT_TRUE(s2.count(label));
          cg2.insert(mesh->dmh->get_point_global_index(index));
        }
      }
    }
  }

  std::set<PetscInt> t1;
  std::set<PetscInt> t2;

  for (auto cx : f1->cells) {
    t1.insert(cx);
  }
  for (auto cx : f2->cells) {
    t2.insert(cx);
  }

  ASSERT_EQ(cg1, t1);
  ASSERT_EQ(cg2, t2);

  ASSERT_EQ(static_cast<std::size_t>(f1->local_dof_count), cg1.size());
  ASSERT_EQ(static_cast<std::size_t>(f2->local_dof_count), cg2.size());

  auto h_dofs1 = f1->get_dofs();
  ASSERT_EQ(h_dofs1.size(), cg1.size());
  auto h_dofs2 = f2->get_dofs();
  ASSERT_EQ(h_dofs2.size(), cg2.size());

  // int index = 0;
  // for (auto cx : f1->cells) {
  //   h_dofs1.at(index++) = cx;
  // }
  // f1->set_dofs(h_dofs1);
  // f1->write_vtkhdf("f1.vtkhdf");

  auto mapper =
      std::make_shared<PetscInterface::DMPlexLocalMapper>(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, mapper);

  ParticleSpec particle_spec{
      ParticleProp(Sym<REAL>("P"), ndim, true),
      ParticleProp(Sym<REAL>("V"), 3),
      ParticleProp(Sym<REAL>("E"), 3),
      ParticleProp(Sym<REAL>("INTERSECTION_POINT"), ndim),
      ParticleProp(Sym<REAL>("NORMAL"), ndim),
      ParticleProp(Sym<INT>("METADATA"), 2),
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  const int N = 10000;
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

  auto lambda_set_dofs_simple = [&](auto &f) {
    auto dofs = f->get_dofs();
    int index = 0;
    for (INT cx : f->cells) {
      dofs.at(index++) = static_cast<REAL>(cx);
    }
    f->set_dofs(dofs);
  };

  lambda_set_dofs_simple(f1);
  lambda_set_dofs_simple(f2);

  particle_loop(
      A,
      [=](auto V) {
        REAL Vmag = 0.0;
        for (int dx = 0; dx < ndim; dx++) {
          Vmag = V.at(dx) * V.at(dx);
        }
        const bool is_zero = Vmag == 0.0;
        const REAL scaling = is_zero ? 0.0 : 1.0 / Kernel::sqrt(Vmag);

        for (int dx = 0; dx < ndim; dx++) {
          V.at(dx) = is_zero ? 1.0 : scaling * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("V")))
      ->execute();

  b2d->pre_integration(A);
  particle_loop(
      A,
      [=](auto P, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) += 1000.0 * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
      ->execute();
  auto groups = b2d->post_integration(A);

  b2d->function_evaluate(groups[1], Sym<REAL>("E"), 1, false, f1);
  b2d->function_evaluate(groups[2], Sym<REAL>("E"), 1, false, f2);

  particle_loop(
      A,
      [=](auto INTERSECTION_POINT, auto NORMAL, auto METADATA) {
        for (int dx = 0; dx < ndim; dx++) {
          INTERSECTION_POINT.at(dx) = -100000.0;
          NORMAL.at(dx) = -100000.0;
          METADATA.at(0) = -1;
          METADATA.at(1) = -1;
        }
      },
      Access::write(Sym<REAL>("INTERSECTION_POINT")),
      Access::write(Sym<REAL>("NORMAL")), Access::write(Sym<INT>("METADATA")))
      ->execute();

  for (auto &gx : {groups[1], groups[2]}) {
    copy_ephemeral_dat_to_particle_dat(
        gx, Sym<REAL>("NESO_PARTICLES_BOUNDARY_INTERSECTION_POINT"),
        Sym<REAL>("INTERSECTION_POINT"));
    copy_ephemeral_dat_to_particle_dat(
        gx, Sym<REAL>("NESO_PARTICLES_BOUNDARY_NORMAL"), Sym<REAL>("NORMAL"));
    copy_ephemeral_dat_to_particle_dat(
        gx, Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA"), Sym<INT>("METADATA"));
  }

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  b2d->free();
  sycl_target->free();
  mesh->free();

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
