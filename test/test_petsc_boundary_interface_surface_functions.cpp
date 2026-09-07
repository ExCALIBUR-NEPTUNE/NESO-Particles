#ifdef NESO_PARTICLES_PETSC

#include "include/test_neso_particles.hpp"
#include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

TEST(PETScBoundary2D, setup_surface_functions_evaluate) {

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
          Vmag += V.at(dx) * V.at(dx);
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

  particle_loop(
      A,
      [=](auto METADATA, auto E) {
        const INT geom_id = METADATA.at(1);
        const REAL geom_id_real = geom_id;

        NESO_KERNEL_ASSERT(geom_id > -1, k_ep);
        NESO_KERNEL_ASSERT(geom_id_real == E.at(1), k_ep);
      },
      Access::read(Sym<INT>("METADATA")), Access::read(Sym<REAL>("E")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  b2d->free();
  sycl_target->free();
  mesh->free();

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

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

void wrapper_mesh_test(
    const int ndim, DM dm,
    std::map<PetscInt, std::vector<PetscInt>> boundary_groups) {

  auto A = particle_loop_common(ndim, dm, 1093);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  const REAL tol = 1.0e-15;
  auto bic = PetscInterface::create_boundary_interaction(sycl_target, mesh,
                                                         boundary_groups, tol);

  std::map<int, PetscInterface::DMPlexFunctionSharedPtr> funcs;
  for (auto &bx : boundary_groups) {
    funcs[bx.first] = bic->create_function(bx.first, "DG", 0);
  }

  particle_loop(
      A,
      [=](auto V) {
        REAL Vmag = 0.0;
        for (int dx = 0; dx < ndim; dx++) {
          Vmag += V.at(dx) * V.at(dx);
        }
        const bool is_zero = Vmag == 0.0;
        const REAL scaling = is_zero ? 0.0 : 1.0 / Kernel::sqrt(Vmag);

        for (int dx = 0; dx < ndim; dx++) {
          V.at(dx) = is_zero ? 1.0 : scaling * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("V")))
      ->execute();

  bic->pre_integration(A);
  particle_loop(
      A,
      [=](auto P, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) += 50.0 * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
      ->execute();
  auto groups = bic->post_integration(A);

  auto face_sets = mesh->dmh->get_face_sets();
  std::set<INT> previously_seen;

  for (auto bx : boundary_groups) {
    auto &func = funcs.at(bx.first);
    std::set<INT> cells;
    for (auto &petsc_index : func->cells) {
      cells.insert(petsc_index);
      ASSERT_EQ(previously_seen.count(petsc_index), 0);
      previously_seen.insert(petsc_index);
    }
    ASSERT_EQ(cells.size(), func->cells.size());

    const int num_cells = cells.size();
    auto h_dofs = func->get_dofs();
    for (int ix = 0; ix < num_cells; ix++) {
      h_dofs.at(ix) = func->cells.at(ix);
    }
    func->set_dofs(h_dofs);
  }

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  for (auto &gx : groups) {
    const auto k_label = gx.first;
    particle_loop(
        gx.second,
        [=](auto U, auto METADATA) {
          NESO_KERNEL_ASSERT(METADATA.at_ephemeral(0) == k_label, k_ep);
          U.at(0) = METADATA.at_ephemeral(1);
          U.at(1) = -1;
        },
        Access::write(Sym<REAL>("U")),
        Access::read(BoundaryInteractionSpecification::intersection_metadata))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  }

  for (auto &gx : groups) {
    bic->function_evaluate(gx.second, Sym<REAL>("U"), 1, false,
                           funcs.at(gx.first));
  }

  for (auto &gx : groups) {
    particle_loop(
        gx.second,
        [=](auto U) { NESO_KERNEL_ASSERT(U.at(0) == U.at(1), k_ep); },
        Access::read(Sym<REAL>("U")))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  }

  INT bound_lower = 0;
  INT bound_upper = 0;
  mesh->dmh->get_global_face_index_bounds(bound_lower, bound_upper);
  const INT num_faces = bound_upper - bound_lower;

  std::vector<REAL> h_contributions(num_faces);
  std::fill(h_contributions.begin(), h_contributions.end(), 0.0);
  BufferDevice<REAL> d_contributions(sycl_target, h_contributions);
  REAL *k_contributions = d_contributions.ptr;

  for (auto gx : groups) {
    particle_loop(
        gx.second,
        [=](auto V, auto METADATA) {
          const INT index = METADATA.at_ephemeral(1);
          const bool valid_index =
              (bound_lower <= index) && (index < bound_upper);
          NESO_KERNEL_ASSERT(valid_index, k_ep);
          if (valid_index) {
            atomic_fetch_add(k_contributions + index - bound_lower, V.at(0));
          }
        },
        Access::read(Sym<REAL>("V")),
        Access::read(BoundaryInteractionSpecification::intersection_metadata))
        ->execute();
    ASSERT_FALSE(ep.get_flag());
  }

  sycl_target->queue
      .memcpy(h_contributions.data(), k_contributions, num_faces * sizeof(REAL))
      .wait_and_throw();

  std::vector<REAL> h_contributions_reduced(num_faces);
  std::fill(h_contributions_reduced.begin(), h_contributions_reduced.end(),
            0.0);

  MPICHK(MPI_Allreduce(h_contributions.data(), h_contributions_reduced.data(),
                       static_cast<int>(num_faces), map_ctype_mpi_type<REAL>(),
                       MPI_SUM, mesh->get_comm()));

  for (auto gx : funcs) {
    int index = 0;
    for (INT &point : gx.second->cells_local) {
      const REAL inverse_volume =
          1.0 / mesh->dmh->get_point_volume(static_cast<PetscInt>(point));

      const auto global_index = gx.second->cells.at(index);

      ASSERT_TRUE((bound_lower <= global_index) &&
                  (global_index < bound_upper));

      const REAL contrib =
          h_contributions_reduced.at(global_index - bound_lower);
      h_contributions_reduced.at(global_index - bound_lower) =
          contrib * inverse_volume;
      index++;
    }
  }

  for (auto gx : groups) {
    bic->function_project_initialise(funcs.at(gx.first));
  }
  for (auto gx : groups) {
    bic->function_project_finalise(funcs.at(gx.first));
  }
  for (auto gx : groups) {
    auto h_dofs = funcs.at(gx.first)->get_dofs();
    for (auto dx : h_dofs) {
      ASSERT_EQ(dx, 0.0);
    }
  }

  int npart_local_A = A->get_npart_local();
  int npart_local_t = 0;
  for (auto gx : groups) {
    bic->function_project(gx.second, Sym<REAL>("V"), 0, false,
                          funcs.at(gx.first));
    npart_local_t += gx.second->get_npart_local();
  }
  ASSERT_EQ(npart_local_t, npart_local_A);

  for (auto func : funcs) {
    auto h_dofs = func.second->get_dofs();
    int index = 0;
    for (auto cellx : func.second->cells) {
      const INT reduced_index = cellx - bound_lower;
      const REAL correct = h_contributions_reduced.at(reduced_index);
      const REAL to_test = h_dofs.at(index);

      ASSERT_TRUE(relative_error(correct, to_test) < 1.0e-11);

      index++;
    }

    // func.second->write_vtkhdf(std::to_string(ndim) + "d_func_" +
    //                           std::to_string(func.first) + ".vtkhdf");
  }

  //{
  //  std::vector<VTK::UnstructuredCell> t;
  //  get_vtk_trajectory_line(
  //    A,
  //    bic->previous_position_sym,
  //    Sym<REAL>("P"),
  //    t
  //  );
  //
  //  VTK::VTKHDF w(std::to_string(ndim) + "t.vtkhdf", mesh->get_comm());
  //  w.write(t);
  //  w.close();
  //}

  bic->free();
  sycl_target->free();
  mesh->free();
}

void wrapper_mesh_mass_test(
    const int ndim, DM dm,
    std::map<PetscInt, std::vector<PetscInt>> boundary_groups) {

  const REAL dt = 0.2;
  const int Nsteps = 100;

  auto A = particle_loop_common(ndim, dm, 10930);
  auto mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
      A->domain->mesh);
  auto sycl_target = A->sycl_target;

  fill(A, Sym<REAL>("MASS"), 1.0);

  const REAL tol = 1.0e-15;
  auto bic = PetscInterface::create_boundary_interaction(sycl_target, mesh,
                                                         boundary_groups, tol);

  std::map<int, PetscInterface::DMPlexFunctionSharedPtr> funcs;
  std::map<int, PetscInterface::DMPlexFunctionSharedPtr> funcs_contrib;
  std::map<int, PetscInterface::DMPlexFunctionSharedPtr> funcs_total;
  for (auto &bx : boundary_groups) {
    funcs[bx.first] = bic->create_function(bx.first, "DG", 0);
    funcs_contrib[bx.first] = bic->create_function(bx.first, "DG", 0);
    funcs_total[bx.first] = bic->create_function(bx.first, "DG", 0);
    funcs_total[bx.first]->fill(0.0);
  }

  particle_loop(
      A,
      [=](auto V) {
        REAL Vmag = 0.0;
        for (int dx = 0; dx < ndim; dx++) {
          Vmag += V.at(dx) * V.at(dx);
        }
        const bool is_zero = Vmag == 0.0;
        const REAL scaling = is_zero ? 0.0 : 1.0 / Kernel::sqrt(Vmag);

        for (int dx = 0; dx < ndim; dx++) {
          V.at(dx) = is_zero ? 1.0 : scaling * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("V")))
      ->execute();

  auto lambda_get_boundary_mass = [&]() -> REAL {
    REAL local_mass = 0.0;
    for (auto &fx : funcs) {
      auto h_dofs = fx.second->get_dofs();
      for (std::size_t dx = 0; dx < h_dofs.size(); dx++) {
        const PetscInt point_index = fx.second->cells_local.at(dx);
        const REAL volume = mesh->dmh->get_point_volume(point_index);
        const REAL dof = h_dofs.at(dx);
        local_mass += volume * dof;
      }
    }
    REAL total_mass = 0.0;
    MPICHK(MPI_Allreduce(&local_mass, &total_mass, 1,
                         map_ctype_mpi_type<REAL>(), MPI_SUM,
                         mesh->get_comm()));
    return total_mass;
  };

  auto lambda_get_total_boundary_mass = [&]() -> REAL {
    REAL local_mass = 0.0;
    for (auto &fx : funcs_total) {
      auto h_dofs = fx.second->get_dofs();
      for (std::size_t dx = 0; dx < h_dofs.size(); dx++) {
        const PetscInt point_index = fx.second->cells_local.at(dx);
        const REAL volume = mesh->dmh->get_point_volume(point_index);
        const REAL dof = h_dofs.at(dx);
        local_mass += volume * dof;
      }
    }
    REAL total_mass = 0.0;
    MPICHK(MPI_Allreduce(&local_mass, &total_mass, 1,
                         map_ctype_mpi_type<REAL>(), MPI_SUM,
                         mesh->get_comm()));
    return total_mass;
  };

  auto ga_mass = std::make_shared<GlobalArray<REAL>>(sycl_target, 1);
  auto lambda_get_particle_mass = [&]() -> REAL {
    ga_mass->fill(0.0);
    particle_loop(
        A, [=](auto MASS, auto GA_MASS) { GA_MASS.add(0, MASS.at(0)); },
        Access::read(Sym<REAL>("MASS")), Access::add(ga_mass))
        ->execute();
    return ga_mass->get().at(0);
  };

  // auto h5part = std::make_shared<H5Part>(
  //     "trajectory_surface_" + std::to_string(ndim) + ".h5part", A,
  //     Sym<REAL>("P"));

  REAL mass_particle_system = lambda_get_particle_mass();
  const REAL mass_total = mass_particle_system;
  REAL mass_boundary_function = lambda_get_boundary_mass();

  for (int stepx = 0; stepx < Nsteps; stepx++) {
    bic->pre_integration(A);
    forward_euler(A, Sym<REAL>("P"), dt, Sym<REAL>("V"));
    auto groups = bic->post_integration(A);

    for (auto &gx : groups) {
      bic->function_project(gx.second, Sym<REAL>("MASS"), 0, false,
                            funcs.at(gx.first));
    }

    for (auto &gx : groups) {
      gx.second->add_ephemeral_dat(Sym<REAL>("EPH_MASS"), 1);
      copy_particle_dat_to_ephemeral_dat(gx.second, Sym<REAL>("MASS"),
                                         Sym<REAL>("EPH_MASS"));

      auto &func_tmp = funcs_contrib.at(gx.first);
      bic->function_project_initialise(func_tmp);
      bic->function_project_contribute(gx.second, Sym<REAL>("MASS"), 0, false,
                                       func_tmp);
      bic->function_project_contribute(gx.second, Sym<REAL>("EPH_MASS"), 0,
                                       true, func_tmp);
      bic->function_project_finalise(func_tmp);
    }

    std::vector<ParticleSubGroupSharedPtr> groups_union;
    for (auto &gx : groups) {
      groups_union.push_back(gx.second);

      auto h_dofs = funcs_total.at(gx.first)->get_dofs();
      auto h_dofs_to_add = funcs.at(gx.first)->get_dofs();

      std::transform(h_dofs.begin(), h_dofs.end(), h_dofs_to_add.begin(),
                     h_dofs.begin(), std::plus<REAL>());

      funcs_total.at(gx.first)->set_dofs(h_dofs);

      // funcs_total.at(gx.first)->write_vtkhdf("trajectory_surface_" +
      //                                        std::to_string(ndim) + "d_" +
      //                                        std::to_string(stepx) +
      //                                        ".vtkhdf");
    }

    // h5part->write();
    A->remove_particles(particle_sub_group_disjoint_union(groups_union));

    mass_particle_system = lambda_get_particle_mass();
    mass_boundary_function += lambda_get_boundary_mass();
    const REAL mass_diff =
        mass_total - mass_boundary_function - mass_particle_system;
    ASSERT_NEAR(mass_diff, 0.0, 1.0e-10);

    const REAL total_mass_to_test = lambda_get_total_boundary_mass();
    ASSERT_TRUE(relative_error(mass_boundary_function, total_mass_to_test) <
                1.0e-14);

    for (auto &gx : groups) {
      auto func_tmp = funcs.at(gx.first);
      auto func_contrib_tmp = funcs_contrib.at(gx.first);
      auto h_dofs_mass = func_tmp->get_dofs();
      auto h_dofs_mass_contrib = func_contrib_tmp->get_dofs();

      for (std::size_t ix = 0; ix < h_dofs_mass.size(); ix++) {
        const REAL correct = h_dofs_mass[ix] * 2.0;
        const REAL to_test = h_dofs_mass_contrib[ix];
        ASSERT_NEAR(correct, to_test, 1.0e-14);
      }
    }
  }

  // h5part->close();

  bic->free();
  sycl_target->free();
  mesh->free();
}

void wrapper_box_mesh_test(const int ndim) {

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

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[0] = {1};
  boundary_groups[1] = {2};
  boundary_groups[2] = {3};
  boundary_groups[3] = {4};
  if (ndim == 3) {
    boundary_groups[4] = {5};
    boundary_groups[5] = {6};
  }

  wrapper_mesh_test(ndim, dm, boundary_groups);
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

void wrapper_box_mesh_mass_test(const int ndim) {

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

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[0] = {1};
  boundary_groups[1] = {2};
  boundary_groups[2] = {3};
  boundary_groups[3] = {4};
  if (ndim == 3) {
    boundary_groups[4] = {5};
    boundary_groups[5] = {6};
  }

  wrapper_mesh_mass_test(ndim, dm, boundary_groups);
  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

} // namespace

TEST(PETScBoundary2D, surface_functions_box_mesh_evaluate_project) {
  wrapper_box_mesh_test(2);
}
TEST(PETScBoundary3D, surface_functions_box_mesh_evaluate_project) {
  wrapper_box_mesh_test(3);
}
TEST(PETScBoundary2D, surface_functions_ref_mesh_evaluate_project) {

  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/reference_all_types_square_0.2.msh");
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  std::vector<int> faces = {100, 200, 300, 400};
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  for (int ix : faces) {
    boundary_groups[ix] = {ix};
  }

  wrapper_mesh_test(2, dm, boundary_groups);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
TEST(PETScBoundary2D, surface_functions_ring_mesh_evaluate_project) {

  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mesh_ring.msh");
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  wrapper_mesh_test(2, dm, boundary_groups);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
TEST(PETScBoundary3D, surface_functions_ref_mesh_evaluate_project) {

  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  std::vector<int> faces = {100, 200, 300, 400, 500, 600};
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  for (int ix : faces) {
    boundary_groups[ix] = {ix};
  }

  wrapper_mesh_test(3, dm, boundary_groups);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
TEST(PETScBoundary2D, surface_functions_box_mesh_mass_conservation) {
  wrapper_box_mesh_mass_test(2);
}
TEST(PETScBoundary3D, surface_functions_box_mesh_mass_conservation) {
  wrapper_box_mesh_mass_test(3);
}
TEST(PETScBoundary2D, surface_functions_ref_mesh_mass_conservation) {

  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/reference_all_types_square_0.2.msh");
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  std::vector<int> faces = {100, 200, 300, 400};
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  for (int ix : faces) {
    boundary_groups[ix] = {ix};
  }

  wrapper_mesh_mass_test(2, dm, boundary_groups);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
TEST(PETScBoundary2D, surface_functions_ring_mesh_mass_conservation) {

  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mesh_ring.msh");
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  boundary_groups[1] = {1, 2, 3, 4};

  wrapper_mesh_mass_test(2, dm, boundary_groups);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}
TEST(PETScBoundary3D, surface_functions_ref_mesh_mass_conservation) {

  std::filesystem::path gmsh_filepath;
  GET_TEST_RESOURCE(gmsh_filepath, "gmsh/mixed_ref_cube_0.8.msh");
  PETSCCHK(PetscInitializeNoArguments());
  DM dm;
  PETSCCHK(DMPlexCreateGmshFromFile(MPI_COMM_WORLD,
                                    gmsh_filepath.generic_string().c_str(),
                                    (PetscBool)1, &dm));
  PetscInterface::generic_distribute(&dm);

  std::vector<int> faces = {100, 200, 300, 400, 500, 600};
  std::map<PetscInt, std::vector<PetscInt>> boundary_groups;
  for (int ix : faces) {
    boundary_groups[ix] = {ix};
  }

  wrapper_mesh_mass_test(3, dm, boundary_groups);

  PETSCCHK(DMDestroy(&dm));
  PETSCCHK(PetscFinalize());
}

#endif
