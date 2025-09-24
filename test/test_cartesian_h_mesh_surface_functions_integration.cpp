#include "include/test_neso_particles.hpp"

namespace {

void mass_conservation_wrapper(const int ndim) {
  const int nx = 16;
  const int ny = 32;
  const int nz = 6;
  const int npart_cell = 4;
  const REAL dt = 0.20;
  const int Nsteps = 100;

  auto [A_t, sycl_target, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);

  auto A = A_t;
  A->add_particle_dat(Sym<REAL>("MASS"), 1);

  particle_loop(
      A, [=](auto MASS) { MASS.at(0) = 1.0; }, Access::write(Sym<REAL>("MASS")))
      ->execute();

  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);

  std::map<int, std::vector<int>> boundary_groups;
  if (ndim == 2) {
    boundary_groups[0] = {0, 1, 2, 3};
  } else {
    boundary_groups[0] = {0, 1, 2, 3, 4, 5};
  }

  auto cti = std::make_shared<CartesianTrajectoryIntersection>(
      sycl_target, mesh, boundary_groups, 1.0e-14);
  cti->prepare_particle_group(A);

  auto func_mass = cti->create_function(0, "DG", 0);
  auto func_mass_contrib = cti->create_function(0, "DG", 0);

  auto lambda_get_boundary_mass = [&]() -> REAL {
    REAL local_mass = 0.0;
    auto h_dofs = func_mass->get_dofs();
    for (auto ix : h_dofs) {
      local_mass += ix;
    }
    REAL total_mass = 0.0;
    MPICHK(MPI_Allreduce(&local_mass, &total_mass, 1,
                         map_ctype_mpi_type<REAL>(), MPI_SUM,
                         mesh->get_comm()));

    return total_mass * std::pow(mesh->cell_width_fine, ndim - 1);
  };

  auto advection_loop = particle_loop(
      A,
      [=](auto P, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          P.at(dx) += dt * V.at(dx);
        }
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")));

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
    cti->pre_integration(A);
    advection_loop->execute();
    auto groups = cti->post_integration(A);

    cti->function_project(groups[0], Sym<REAL>("MASS"), 0, false, func_mass);

    groups.at(0)->add_ephemeral_dat(Sym<REAL>("EPH_MASS"), 1);
    particle_loop(
        groups.at(0),
        [=](auto MASS, auto EPH_MASS) {
          EPH_MASS.at_ephemeral(0) = MASS.at(0);
        },
        Access::read(Sym<REAL>("MASS")), Access::write(Sym<REAL>("EPH_MASS")))
        ->execute();

    cti->function_project_initialise(func_mass_contrib);
    cti->function_project_contribute(groups[0], Sym<REAL>("MASS"), 0, false,
                                     func_mass_contrib);
    cti->function_project_contribute(groups[0], Sym<REAL>("EPH_MASS"), 0, true,
                                     func_mass_contrib);
    cti->function_project_finalise(func_mass_contrib);

    A->remove_particles(groups.at(0));

    mass_particle_system = lambda_get_particle_mass();
    mass_boundary_function += lambda_get_boundary_mass();
    const REAL mass_diff =
        mass_total - mass_boundary_function - mass_particle_system;
    ASSERT_NEAR(mass_diff, 0.0, 1.0e-10);

    auto h_dofs_mass = func_mass->get_dofs();
    auto h_dofs_mass_contrib = func_mass_contrib->get_dofs();

    for (std::size_t ix = 0; ix < h_dofs_mass.size(); ix++) {
      const REAL correct = h_dofs_mass[ix] * 2.0;
      const REAL to_test = h_dofs_mass_contrib[ix];
      ASSERT_NEAR(correct, to_test, 1.0e-14);
    }

    // func_mass->write_vtkhdf("trajectory_surface_" + std::to_string(ndim) +
    //                         "d_" + std::to_string(stepx) + ".vtkhdf");

    // h5part->write();
  }

  // h5part->close();

  cti->free();
  mesh->free();
}

void reflection_evaluation_wrapper(const int ndim) {
  const int nx = 16;
  const int ny = 32;
  const int nz = 6;
  const int npart_cell = 4;
  const REAL dt = 1.00;
  const int Nsteps = 500;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  const int rank = sycl_target->comm_pair.rank_parent;
  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);

  A->clear();
  if (rank == 0) {
    ParticleSet initial_distribution(1, A->get_particle_spec());
    std::vector<REAL> positions(ndim);
    mesh->get_point_in_subdomain(positions.data());
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][0][dimx] = positions[dimx];
      initial_distribution[Sym<REAL>("V")][0][dimx] = 1.0;
    }
    A->add_particles_local(initial_distribution);
  }

  parallel_advection_initialisation(A, 16);
  A->add_particle_dat(Sym<REAL>("TSP"), 2);
  A->add_particle_dat(Sym<REAL>("Q"), 1);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  std::map<int, std::vector<int>> boundary_groups;
  if (ndim == 2) {
    boundary_groups[0] = {0, 1, 2, 3};
  } else {
    boundary_groups[0] = {0, 1, 2, 3, 4, 5};
  }

  auto cti = std::make_shared<CartesianTrajectoryIntersection>(
      sycl_target, mesh, boundary_groups, 1.0e-10);
  cti->prepare_particle_group(A);

  std::map<int, CartesianHMeshFunctionSharedPtr> funcs;
  for (auto &gx : boundary_groups) {
    auto func_gid = cti->create_function(gx.first, "DG", 0);
    auto h_dofs = func_gid->get_dofs();
    int index = 0;
    for (INT cx : func_gid->cells) {
      h_dofs.at(index++) = cx;
    }
    func_gid->set_dofs(h_dofs);
    funcs[gx.first] = func_gid;
  }

  auto reflection = std::make_shared<BoundaryReflection>(ndim, 1.0e-8);

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    auto sub_groups = cti->post_integration(aa);
    for (auto &gx : sub_groups) {
      reflection->execute(gx.second, Sym<REAL>("P"), Sym<REAL>("V"),
                          Sym<REAL>("TSP"), cti->previous_position_sym);

      cti->function_evaluate(gx.second, Sym<REAL>("Q"), 0, false,
                             funcs.at(gx.first));

      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      particle_loop(
          gx.second,
          [=](auto Q, auto META) {
            const REAL geom_id_real = META.at_ephemeral(1);
            NESO_KERNEL_ASSERT(Kernel::abs(Q.at(0) - geom_id_real) < 1.0e-14,
                               k_ep);
          },
          Access::read(Sym<REAL>("Q")),
          Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")))
          ->execute();
      ASSERT_FALSE(ep.get_flag());
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
  auto lambda_pre_advection = [&](auto aa) { cti->pre_integration(aa); };
  auto lambda_find_partial_moves = [&](auto aa) {
    return static_particle_sub_group(
        aa, [=](auto TSP) { return TSP.at(0) < dt; },
        Access::read(Sym<REAL>("TSP")));
  };
  auto lambda_partial_moves_remaining = [&](auto aa) -> bool {
    const int size = get_npart_global(aa);
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

  auto loop_perturb = particle_loop(
      A,
      [=](auto V, auto P) {
        for (int dx = 0; dx < ndim; dx++) {
          V.at(dx) += 0.1 * P.at(dx);
        }
      },
      Access::write(Sym<REAL>("V")), Access::read(Sym<REAL>("P")));

  // funcs[0]->write_vtkhdf("func_gid_" + std::to_string(ndim) +
  //                         "d.vtkhdf");

  // uncomment to write a trajectory
  // H5Part h5part("traj_reflection_surface_evaluation.h5part", A,
  // Sym<REAL>("P"),
  //              Sym<REAL>("V"), Sym<REAL>("Q"));
  for (int stepx = 0; stepx < Nsteps; stepx++) {
    lambda_apply_timestep(static_particle_sub_group(A));
    // uncomment to write a trajectory
    // h5part.write();

    A->hybrid_move();
    ccb->execute();
    A->cell_move();
    loop_perturb->execute();
  }

  // uncomment to write a trajectory
  // h5part.close();

  cti->free();
  mesh->free();
}

} // namespace

TEST(CartesianHMesh, surface_functions_mass_conservation_2d) {
  mass_conservation_wrapper(2);
}

TEST(CartesianHMesh, surface_functions_mass_conservation_3d) {
  mass_conservation_wrapper(3);
}

TEST(CartesianHMesh, surface_functions_reflection_evaluation_2d) {
  reflection_evaluation_wrapper(2);
}

TEST(CartesianHMesh, surface_functions_reflection_evaluation_3d) {
  reflection_evaluation_wrapper(3);
}
