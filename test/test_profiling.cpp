#include "include/test_neso_particles.hpp"

TEST(Profiling, per_particle_average) {

  std::size_t key1 = 1;
  std::size_t key2 = 2;

  auto r0 =
      ProfileRegion("key1", "key2", ProfileRegionType::PER_PARTICLE_TIMING);

  ASSERT_EQ(r0.profile_region_type_coeff, 0.0);

  r0.end(key1, key2, 1024);

  ASSERT_EQ(r0.profile_region_type_coeff, static_cast<double>(1024));

  ProfileMap profile_map;
  profile_map.enable();
  auto r1 = profile_map.start_region("key1", "key2",
                                     ProfileRegionType::PER_PARTICLE_TIMING);

  profile_map.end_region(r1, key1, key2, 1024);
  ASSERT_EQ(r1.value().profile_region_type_coeff, static_cast<double>(1024));
}

TEST(Profiling, per_cell_cost) {
  const int ndim = 2;
  const int nx = 32;
  const int ny = 32;
  const int nz = 6;
  const int npart_cell = 4;
  const REAL dt = 0.10;
  const int Nsteps = 400;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  const int rank = sycl_target->comm_pair.rank_parent;
  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

  const int num_cell_bins = 1600;
  const int num_particles_per_cell = 1000;
  const int cell_count = mesh->get_cell_count();

  const int N = num_particles_per_cell * cell_count;
  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, A->get_particle_spec());

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int ix = 0; ix < num_particles_per_cell; ix++) {
      const int px = ix + cellx * num_particles_per_cell;
      for (int dimx = 0; dimx < ndim; dimx++) {
        initial_distribution.at(Sym<REAL>("P"), px, dimx) = positions[dimx][px];
      }
      for (int dimx = 0; dimx < 3; dimx++) {
        initial_distribution.at(Sym<REAL>("V"), px, dimx) =
            velocities[dimx][px];
      }
      initial_distribution.at(Sym<INT>("CELL_ID"), px, 0) = cellx;
    }
  }

  A->add_particles_local(initial_distribution);
  parallel_advection_initialisation(A, 16);
  A->add_particle_dat(Sym<REAL>("TSP"), 2);
  A->add_particle_dat(Sym<REAL>("Q"), 1);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  //// uncomment to write a trajectory
  // H5Part h5part("traj_reflection_surface_evaluation.h5part", A,
  // Sym<REAL>("P"),
  //              Sym<REAL>("V"), Sym<REAL>("Q"));
  //
  //{
  //   auto to_remove = particle_sub_group(
  //     A,
  //     [=](auto INDEX){
  //       const int cutoff = INDEX.cell % num_cell_bins;
  //       return INDEX.layer >= cutoff;
  //     },
  //     Access::read(ParticleLoopIndex{})
  //   );
  //   A->remove_particles(to_remove);
  // }
  //
  // for(int cellx=0 ; cellx<cell_count ; cellx++){
  //   ASSERT_EQ(A->get_npart_cell(cellx), cellx % num_cell_bins);
  // }
  //
  // A->domain->profiling_cell_costs.enable();
  // sycl_target->profile_map.enable();
  // for (int stepx = 0; stepx < Nsteps; stepx++) {
  //   nprint_variable("stepx");
  //   // uncomment to write a trajectory
  //   h5part.write();
  //
  //   particle_loop(
  //     A,
  //     [=](auto P, auto V){
  //       for(int dx=0 ; dx<ndim ; dx++){
  //         P.at(dx) += dt * V.at(dx);
  //       }
  //     },
  //     Access::write(Sym<REAL>("P")),
  //     Access::read(Sym<REAL>("V"))
  //   )->execute();
  //
  //   //A->hybrid_move();
  //   //ccb->execute();
  //   //A->cell_move();
  // }
  //
  // std::vector<REAL> h_cell_costs;
  // A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);
  // nprint(h_cell_costs);
  //
  // auto cdc_test =
  //     std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 1, 1);
  // cdc_test->fill(0.0);
  //
  // particle_loop(
  //   A,
  //   [=](auto CDC){
  //     CDC.fetch_add(0, 0, 1);
  //   },
  //   Access::add(cdc_test)
  //)->execute();
  //
  //
  // auto cell_data = mesh->get_vtk_cell_data();
  // auto test_cdc_data = cdc_test->get_all_cells();
  //
  // for(int cellx=0 ; cellx<cell_count ; cellx++){
  //
  //   ASSERT_EQ(A->get_npart_cell(cellx), cellx % num_cell_bins);
  //
  //   cell_data.at(cellx).cell_data["cost"] = h_cell_costs.at(cellx);
  //   cell_data.at(cellx).cell_data["test"] = test_cdc_data.at(cellx)->at(0,0);
  // }
  //
  // VTK::VTKHDF vtkhdf("cell_costs.vtkhdf", MPI_COMM_WORLD);
  // vtkhdf.write(cell_data);
  // vtkhdf.close();
  //
  //// uncomment to write a trajectory
  // h5part.close();

  std::map<int, std::vector<int>> boundary_groups;
  if (ndim == 2) {
    boundary_groups[0] = {0, 1, 2, 3};
  } else {
    boundary_groups[0] = {0, 1, 2, 3, 4, 5};
  }

  auto cti = std::make_shared<CartesianTrajectoryIntersection>(
      sycl_target, mesh, boundary_groups, 1.0e-10);
  cti->prepare_particle_group(A);

  auto reflection = std::make_shared<BoundaryReflection>(ndim, 1.0e-8);

  auto lambda_apply_boundary_conditions = [&](auto aa) {
    auto sub_groups = cti->post_integration(aa);
    for (auto &gx : sub_groups) {
      reflection->execute(gx.second, Sym<REAL>("P"), Sym<REAL>("V"),
                          Sym<REAL>("TSP"), cti->previous_position_sym);
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

  // funcs[0]->write_vtkhdf("func_gid_" + std::to_string(ndim) +
  //                         "d.vtkhdf");

  // uncomment to write a trajectory
  // H5Part h5part("traj_reflection_surface_evaluation.h5part", A,
  // Sym<REAL>("P"),
  //              Sym<REAL>("V"), Sym<REAL>("Q"));

  A->domain->profiling_cell_costs.enable();
  sycl_target->profile_map.enable();
  for (int stepx = 0; stepx < Nsteps; stepx++) {
    nprint_variable("stepx");
    lambda_apply_timestep(static_particle_sub_group(A));
    // uncomment to write a trajectory
    // h5part.write();

    A->hybrid_move();
    ccb->execute();
    A->cell_move();
  }

  std::vector<REAL> h_cell_costs;
  A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);
  nprint(h_cell_costs);

  auto cell_data = mesh->get_vtk_cell_data();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    cell_data.at(cellx).cell_data["cost"] = h_cell_costs.at(cellx);
  }

  VTK::VTKHDF vtkhdf("cell_costs.vtkhdf", MPI_COMM_WORLD);
  vtkhdf.write(cell_data);
  vtkhdf.close();

  // uncomment to write a trajectory
  // h5part.close();

  cti->free();
  sycl_target->free();
  mesh->free();
}
