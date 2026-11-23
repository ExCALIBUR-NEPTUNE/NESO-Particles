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
  const int nx = 16;
  const int ny = 16;
  const int nz = 6;
  const int npart_cell = 4;
  const REAL dt = 0.02;
  const int Nsteps = 50;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  const int rank = sycl_target->comm_pair.rank_parent;
  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

  const int num_particles_per_cell = 400;
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

  A->clear();
  A->add_particles_local(initial_distribution);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  A->domain->profiling_cell_costs.enable();
  sycl_target->profile_map.enable();

  for (int stepx = 0; stepx < Nsteps; stepx++) {
    particle_loop(
        A,
        [=](auto P, auto V) {
          for (int dx = 0; dx < ndim; dx++) {
            P.at(dx) += dt * V.at(dx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
  }

  std::vector<REAL> h_cell_costs;
  A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const auto cell_cost_tmp = h_cell_costs.at(cellx);
    EXPECT_TRUE(cell_cost_tmp > 0.0);
  }

  A->domain->profiling_cell_costs.zero();
  A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const auto cell_cost_tmp = h_cell_costs.at(cellx);
    EXPECT_TRUE(cell_cost_tmp == 0.0);
  }

  if (cell_count > 3) {
    for (int stepx = 0; stepx < Nsteps; stepx++) {
      particle_loop(
          A,
          [=](auto P, auto V) {
            for (int dx = 0; dx < ndim; dx++) {
              P.at(dx) += dt * V.at(dx);
            }
          },
          Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
          ->execute(0, cell_count - 3);
    }

    A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);

    for (int cellx = 0; cellx < cell_count - 3; cellx++) {
      const auto cell_cost_tmp = h_cell_costs.at(cellx);
      EXPECT_TRUE(cell_cost_tmp > 0.0);
    }

    for (int cellx = cell_count - 3; cellx < cell_count; cellx++) {
      const auto cell_cost_tmp = h_cell_costs.at(cellx);
      EXPECT_TRUE(cell_cost_tmp == 0.0);
    }
  }

  auto aa = static_particle_sub_group(
      A, [=](auto INDEX) { return INDEX.cell % 2 == 0; },
      Access::read(ParticleLoopIndex{}));

  A->domain->profiling_cell_costs.zero();
  A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const auto cell_cost_tmp = h_cell_costs.at(cellx);
    EXPECT_TRUE(cell_cost_tmp == 0.0);
  }

  for (int stepx = 0; stepx < Nsteps; stepx++) {
    particle_loop(
        aa,
        [=](auto P, auto V) {
          for (int dx = 0; dx < ndim; dx++) {
            P.at(dx) += dt * V.at(dx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
  }
  A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const auto cell_cost_tmp = h_cell_costs.at(cellx);
    if (cellx % 2 == 0) {
      EXPECT_TRUE(cell_cost_tmp > 0.0);
    } else {
      EXPECT_TRUE(cell_cost_tmp == 0.0);
    }
  }

  A->domain->profiling_cell_costs.zero();
  if (cell_count > 3) {
    for (int stepx = 0; stepx < Nsteps; stepx++) {
      particle_loop(
          aa,
          [=](auto P, auto V) {
            for (int dx = 0; dx < ndim; dx++) {
              P.at(dx) += dt * V.at(dx);
            }
          },
          Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
          ->execute(0, cell_count - 3);
    }

    A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);

    for (int cellx = 0; cellx < cell_count - 3; cellx++) {
      const auto cell_cost_tmp = h_cell_costs.at(cellx);
      if (cellx % 2 == 0) {
        EXPECT_TRUE(cell_cost_tmp > 0.0);
      } else {
        EXPECT_TRUE(cell_cost_tmp == 0.0);
      }
    }

    for (int cellx = cell_count - 3; cellx < cell_count; cellx++) {
      const auto cell_cost_tmp = h_cell_costs.at(cellx);
      EXPECT_TRUE(cell_cost_tmp == 0.0);
    }
  }

  A->domain->profiling_cell_costs.zero();
  A->domain->profiling_cell_costs.disable();

  for (int stepx = 0; stepx < Nsteps; stepx++) {
    particle_loop(
        A,
        [=](auto P, auto V) {
          for (int dx = 0; dx < ndim; dx++) {
            P.at(dx) += dt * V.at(dx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
  }

  A->domain->profiling_cell_costs.get_cell_costs(h_cell_costs);
  EXPECT_EQ(h_cell_costs.size(), 0);

  sycl_target->free();
  mesh->free();
}
