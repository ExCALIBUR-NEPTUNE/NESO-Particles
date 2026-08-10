#include "include/test_neso_particles.hpp"

TEST(DSMC, all_to_all_looping_host) {

  const int workgroup_size = 4;

  for (int num_particles : {0, 1, 3, 4, 7, 8, 31}) {

    std::vector<int> seen_entries(num_particles * num_particles);
    std::fill(seen_entries.begin(), seen_entries.end(), 0);
    std::vector<int> num_sync_calls(workgroup_size);
    std::fill(num_sync_calls.begin(), num_sync_calls.end(), 0);

    for (int workgroup_item = 0; workgroup_item < workgroup_size;
         workgroup_item++) {
      auto loop_context =
          DSMC::CellwiseAllToAll(workgroup_size, workgroup_item);

      loop_context.apply(
          num_particles,
          [&](auto i, auto j) {
            seen_entries.at(i + num_particles * j)++;
            seen_entries.at(j + num_particles * i)++;
          },
          [&]() { num_sync_calls.at(workgroup_item)++; });
    }

    for (int ix = 0; ix < workgroup_size; ix++) {
      ASSERT_EQ(num_sync_calls.at(ix), num_sync_calls.at(0));
    }

    for (int rowx = 0; rowx < num_particles; rowx++) {
      for (int colx = 0; colx < num_particles; colx++) {
        ASSERT_EQ(seen_entries.at(rowx + num_particles * colx),
                  rowx == colx ? 0 : 1);
      }
    }
  }
}

TEST(DSMC, all_to_all_looping_device) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

#ifdef __ACPP__
  const int cpu_workgroup_size = 1;
  const bool use_barrier = sycl_target->device.is_cpu() ? false : true;
#else
  const int cpu_workgroup_size = 32;
  constexpr bool use_barrier = true;
#endif

  const int workgroup_size =
      sycl_target->device.is_cpu() ? cpu_workgroup_size : 256;

  for (int num_particles : {0, 1, 3, 4, 7, 8, 31, 255, 1023, 1024}) {

    std::vector<int> seen_entries(std::max(num_particles * num_particles, 1));
    std::fill(seen_entries.begin(), seen_entries.end(), 0);
    BufferDevice<int> d_seen_entries(sycl_target, seen_entries);
    auto k_seen_entries = d_seen_entries.ptr;

    sycl_target->queue
        .parallel_for(sycl::nd_range<1>(sycl::range<1>(workgroup_size),
                                        sycl::range<1>(workgroup_size)),
                      [=](sycl::nd_item<1> idx) {
                        DSMC::CellwiseAllToAll loop_context(
                            idx.get_local_range(0), idx.get_local_id(0));

                        loop_context.apply(
                            num_particles,
                            [&](auto i, auto j) {
                              k_seen_entries[i + num_particles * j]++;
                              k_seen_entries[j + num_particles * i]++;
                            },
                            [=]() {
                              if (use_barrier) {
                                sycl::group_barrier(idx.get_group());
                              }
                            });
                      })
        .wait_and_throw();

    seen_entries = d_seen_entries.get();

    for (int rowx = 0; rowx < num_particles; rowx++) {
      for (int colx = 0; colx < num_particles; colx++) {
        ASSERT_EQ(seen_entries.at(rowx + num_particles * colx),
                  rowx == colx ? 0 : 1);
      }
    }
  }

  sycl_target->free();
}

TEST(DSMC, ntc_pair_generation_aa) {

  const int max_num_pairs_to_sample = 1003;
  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 32;
  const int ny = 16;
  const int nz = 48;

  auto [A, sycl_target_t, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto sycl_target = sycl_target_t;
  A->clear();

  ParticleSpec particle_spec{ParticleProp(Sym<INT>("CELL_ID"), 1, true)};

  int N = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    N += cellx;
  }

  ParticleSet distribution(N, particle_spec);

  int ix = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int px = 0; px < cellx; px++) {
      distribution[Sym<INT>("CELL_ID")][ix + px][0] = cellx;
    }
    ix += cellx;
  }
  A->add_particles_local(distribution);

  auto aa = particle_sub_group(A, [=]() { return true; });
  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(A->get_npart_cell(cellx), cellx);
    ASSERT_EQ(aa->get_npart_cell(cellx), cellx);
  }

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng(34234 + rank);
  std::uniform_real_distribution<REAL> dist{
      std::uniform_real_distribution<REAL>(0.0, 1.0)};
  auto lambda_sampler = [&]() -> REAL { return dist(rng); };

  auto rng_function =
      std::make_shared<HostRNGGenerationFunction<REAL>>(lambda_sampler);

  auto pair_sampler_ntc = std::make_shared<DSMC::PairSamplerNTC>(
      sycl_target, cell_count, rng_function);

  std::vector<int> num_pairs(cell_count);
  std::fill(num_pairs.begin(), num_pairs.end(), 0);

  pair_sampler_ntc->sample(aa, aa, num_pairs);

  auto d_pair_list = pair_sampler_ntc->get_pair_list();
  auto h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);

  const std::size_t default_local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  ASSERT_EQ(d_pair_list.cell_count, cell_count);
  ASSERT_EQ(d_pair_list.pair_count, 0);
  ASSERT_EQ(d_pair_list.block_size, default_local_size);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(std::get<0>(h_pair_list[cellx]).size(), 0);
    ASSERT_EQ(std::get<1>(h_pair_list[cellx]).size(), 0);
    ASSERT_EQ(std::get<2>(h_pair_list[cellx]).size(), 0);
  }

  sycl_target->parameters->set("LOOP_LOCAL_SIZE",
                               std::make_shared<SizeTParameter>(8));

  pair_sampler_ntc = std::make_shared<DSMC::PairSamplerNTC>(
      sycl_target, cell_count, rng_function);

  int expected_num_pairs = 0;
  for (int cellx = 2; cellx < cell_count; cellx++) {
    const int n = max_num_pairs_to_sample + cellx % 8;
    num_pairs[cellx] = n;
    expected_num_pairs += n;
  }

  pair_sampler_ntc->sample(aa, aa, num_pairs);
  ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));
  d_pair_list = pair_sampler_ntc->get_pair_list();
  h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);

  ASSERT_EQ(d_pair_list.cell_count, cell_count);
  ASSERT_EQ(d_pair_list.pair_count, expected_num_pairs);
  ASSERT_EQ(d_pair_list.block_size, 8);

  int test_total_num_pairs = 0;
  for (int cellx = 2; cellx < cell_count; cellx++) {
    const int n = max_num_pairs_to_sample + cellx % 8;
    const int npart_cell = aa->get_npart_cell(cellx);
    const int num_pairs_test = n;

    ASSERT_EQ(std::get<0>(h_pair_list[cellx]).size(), num_pairs_test);
    ASSERT_EQ(std::get<1>(h_pair_list[cellx]).size(), num_pairs_test);
    ASSERT_EQ(std::get<2>(h_pair_list[cellx]).size(), num_pairs_test);

    for (int ix = 0; ix < num_pairs_test; ix++) {
      ASSERT_NE(std::get<0>(h_pair_list[cellx])[ix],
                std::get<1>(h_pair_list[cellx])[ix]);
      ASSERT_TRUE(std::get<0>(h_pair_list[cellx])[ix] >= 0);
      ASSERT_TRUE(std::get<1>(h_pair_list[cellx])[ix] >= 0);
      ASSERT_TRUE(std::get<0>(h_pair_list[cellx])[ix] < npart_cell);
      ASSERT_TRUE(std::get<1>(h_pair_list[cellx])[ix] < npart_cell);
      ASSERT_TRUE(std::get<2>(h_pair_list[cellx])[ix] >= 0);
    }
    test_total_num_pairs += num_pairs_test;
  }
  ASSERT_EQ(test_total_num_pairs, expected_num_pairs);

  for (int cellx = 2; cellx < cell_count; cellx++) {
    const int n = max_num_pairs_to_sample;
    num_pairs[cellx] = n;
  }
  const int max_cell = std::min(8, cell_count);
  for (int cellx = max_cell; cellx < cell_count; cellx++) {
    num_pairs[cellx] = 0;
  }

  std::map<int, std::vector<int>> map_cell_adj_matrix;

  for (int cellx = 2; cellx < max_cell; cellx++) {
    const int npart_cell = aa->get_npart_cell(cellx);
    map_cell_adj_matrix[cellx].resize(npart_cell * npart_cell);
    std::fill(map_cell_adj_matrix[cellx].begin(),
              map_cell_adj_matrix[cellx].end(), 0);
  }

  const int num_steps = 100;
  for (int sx = 0; sx < num_steps; sx++) {
    pair_sampler_ntc->sample(aa, aa, num_pairs);
    ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));
    h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);
    for (int cellx = 2; cellx < max_cell; cellx++) {

      const int npart_cell = aa->get_npart_cell(cellx);
      const int num_pairs_test =
          static_cast<int>(std::get<0>(h_pair_list[cellx]).size());

      for (int ix = 0; ix < num_pairs_test; ix++) {
        const int i = std::get<0>(h_pair_list[cellx])[ix];
        const int j = std::get<1>(h_pair_list[cellx])[ix];
        map_cell_adj_matrix[cellx][j * npart_cell + i]++;
      }
    }
  }

  for (int cellx = 2; cellx < max_cell; cellx++) {

    const int npart_cell = aa->get_npart_cell(cellx);
    const int total_num_pairs = npart_cell * npart_cell - npart_cell;
    const REAL average_pair_count =
        static_cast<REAL>(max_num_pairs_to_sample * num_steps) /
        static_cast<REAL>(total_num_pairs);

    for (int jx = 0; jx < npart_cell; jx++) {
      for (int ix = 0; ix < npart_cell; ix++) {
        const int pair_count = map_cell_adj_matrix[cellx][jx * npart_cell + ix];
        if (ix != jx) {
          ASSERT_TRUE(relative_error(average_pair_count,
                                     static_cast<REAL>(pair_count)) < 0.1);
        } else {
          ASSERT_EQ(pair_count, 0);
        }
      }
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(DSMC, ntc_pair_generation_aa_bb) {

  const int max_num_pairs_to_sample = 1001;
  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 32;
  const int ny = 16;
  const int nz = 48;

  auto [A, sycl_target_t, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto sycl_target = sycl_target_t;
  A->clear();

  ParticleSpec particle_spec{
      ParticleProp(Sym<INT>("CELL_ID"), 1, true),
      ParticleProp(Sym<INT>("ID"), 1),
  };

  int N = 0;
  for (int cellx = 2; cellx < cell_count; cellx++) {
    N += cellx * 2 - 1;
  }

  ParticleSet distribution(N, particle_spec);

  int ix = 0;
  for (int cellx = 2; cellx < cell_count; cellx++) {
    for (int px = 0; px < cellx; px++) {
      distribution[Sym<INT>("CELL_ID")][ix + px][0] = cellx;
      distribution[Sym<INT>("ID")][ix + px][0] = 0;
    }
    for (int px = 0; px < cellx - 1; px++) {
      distribution[Sym<INT>("CELL_ID")][ix + px + cellx][0] = cellx;
      distribution[Sym<INT>("ID")][ix + px + cellx][0] = 1;
    }

    ix += cellx * 2 - 1;
  }
  A->add_particles_local(distribution);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) == 0; }, Access::read(Sym<INT>("ID")));
  auto bb = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) == 1; }, Access::read(Sym<INT>("ID")));

  for (int cellx = 2; cellx < cell_count; cellx++) {
    ASSERT_EQ(A->get_npart_cell(cellx), cellx * 2 - 1);
    ASSERT_EQ(aa->get_npart_cell(cellx), cellx);
    ASSERT_EQ(bb->get_npart_cell(cellx), cellx - 1);
  }

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng(34234 + rank);
  std::uniform_real_distribution<REAL> dist{
      std::uniform_real_distribution<REAL>(0.0, 1.0)};
  auto lambda_sampler = [&]() -> REAL { return dist(rng); };

  auto rng_function =
      std::make_shared<HostRNGGenerationFunction<REAL>>(lambda_sampler);

  auto pair_sampler_ntc = std::make_shared<DSMC::PairSamplerNTC>(
      sycl_target, cell_count, rng_function);

  std::vector<int> num_pairs(cell_count);
  std::fill(num_pairs.begin(), num_pairs.end(), 0);

  pair_sampler_ntc->sample(aa, bb, num_pairs);

  auto d_pair_list = pair_sampler_ntc->get_pair_list();
  auto h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);

  const std::size_t default_local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  ASSERT_EQ(d_pair_list.cell_count, cell_count);
  ASSERT_EQ(d_pair_list.pair_count, 0);
  ASSERT_EQ(d_pair_list.block_size, default_local_size);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(std::get<0>(h_pair_list[cellx]).size(), 0);
    ASSERT_EQ(std::get<1>(h_pair_list[cellx]).size(), 0);
    ASSERT_EQ(std::get<2>(h_pair_list[cellx]).size(), 0);
  }

  sycl_target->parameters->set("LOOP_LOCAL_SIZE",
                               std::make_shared<SizeTParameter>(8));

  pair_sampler_ntc = std::make_shared<DSMC::PairSamplerNTC>(
      sycl_target, cell_count, rng_function);

  int expected_num_pairs = 0;
  for (int cellx = 2; cellx < cell_count; cellx++) {
    num_pairs[cellx] = max_num_pairs_to_sample;
    expected_num_pairs += max_num_pairs_to_sample;
  }

  pair_sampler_ntc->sample(aa, bb, num_pairs);

  d_pair_list = pair_sampler_ntc->get_pair_list();
  h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);

  ASSERT_EQ(d_pair_list.cell_count, cell_count);
  ASSERT_EQ(d_pair_list.pair_count, expected_num_pairs);
  ASSERT_EQ(d_pair_list.block_size, 8);

  int test_total_num_pairs = 0;
  for (int cellx = 2; cellx < cell_count; cellx++) {
    const int npart_cell = A->get_npart_cell(cellx);
    const int num_pairs_test =
        static_cast<int>(std::get<0>(h_pair_list[cellx]).size());

    auto ID = A->get_cell(Sym<INT>("ID"), cellx);

    ASSERT_EQ(std::get<0>(h_pair_list[cellx]).size(), num_pairs_test);
    ASSERT_EQ(std::get<1>(h_pair_list[cellx]).size(), num_pairs_test);
    ASSERT_EQ(std::get<2>(h_pair_list[cellx]).size(), num_pairs_test);

    for (int ix = 0; ix < num_pairs_test; ix++) {
      const int i = std::get<0>(h_pair_list[cellx])[ix];
      const int j = std::get<1>(h_pair_list[cellx])[ix];

      ASSERT_EQ(ID->at(i, 0), 0);
      ASSERT_EQ(ID->at(j, 0), 1);

      ASSERT_NE(i, j);
      ASSERT_TRUE(i >= 0);
      ASSERT_TRUE(j >= 0);
      ASSERT_TRUE(i < npart_cell);
      ASSERT_TRUE(j < npart_cell);
      ASSERT_TRUE(std::get<2>(h_pair_list[cellx])[ix] >= 0);
    }
    test_total_num_pairs += num_pairs_test;
  }
  ASSERT_EQ(test_total_num_pairs, expected_num_pairs);

  const int max_cell = std::min(8, cell_count);
  for (int cellx = max_cell; cellx < cell_count; cellx++) {
    num_pairs[cellx] = 0;
  }

  std::map<int, std::vector<int>> map_cell_adj_matrix;

  for (int cellx = 2; cellx < max_cell; cellx++) {
    const int npart_cell = A->get_npart_cell(cellx);
    map_cell_adj_matrix[cellx].resize(npart_cell * npart_cell);
    std::fill(map_cell_adj_matrix[cellx].begin(),
              map_cell_adj_matrix[cellx].end(), 0);
  }

  const int num_steps = 100;
  for (int sx = 0; sx < num_steps; sx++) {
    pair_sampler_ntc->sample(aa, bb, num_pairs);
    ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));
    h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);
    for (int cellx = 2; cellx < max_cell; cellx++) {

      const int npart_cell = A->get_npart_cell(cellx);
      const int num_pairs_test =
          static_cast<int>(std::get<0>(h_pair_list[cellx]).size());

      for (int ix = 0; ix < num_pairs_test; ix++) {
        const int i = std::get<0>(h_pair_list[cellx])[ix];
        const int j = std::get<1>(h_pair_list[cellx])[ix];
        map_cell_adj_matrix[cellx][j * npart_cell + i]++;
      }
    }
  }

  for (int cellx = 2; cellx < max_cell; cellx++) {

    const int npart_cell = A->get_npart_cell(cellx);
    const int npart_cell_aa = aa->get_npart_cell(cellx);
    const int npart_cell_bb = bb->get_npart_cell(cellx);
    const int total_num_pairs = npart_cell_aa * npart_cell_bb;
    const REAL average_pair_count =
        static_cast<REAL>(max_num_pairs_to_sample * num_steps) /
        static_cast<REAL>(total_num_pairs);

    auto ID = A->get_cell(Sym<INT>("ID"), cellx);

    for (int jx = 0; jx < npart_cell; jx++) {
      for (int ix = 0; ix < npart_cell; ix++) {
        const int pair_count = map_cell_adj_matrix[cellx][jx * npart_cell + ix];

        const bool is_valid_pair = (ID->at(ix, 0) == 0) && (ID->at(jx, 0) == 1);
        if (is_valid_pair) {
          ASSERT_TRUE(relative_error(average_pair_count,
                                     static_cast<REAL>(pair_count)) < 0.1);
        } else {
          ASSERT_EQ(pair_count, 0);
        }
      }
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}

namespace {

class TestCollisionCellRateReduction : public DSMC::CollisionCellRateReduction {
public:
  template <typename... ARGS>
  TestCollisionCellRateReduction(ARGS... args)
      : DSMC::CollisionCellRateReduction(args...) {}

  MAKE_GETTER_METHOD(num_contributors)
  MAKE_GETTER_METHOD(d_num_collision_cells)
  MAKE_GETTER_METHOD(d_accumulation_max)
  MAKE_GETTER_METHOD(num_entries)
  MAKE_GETTER_METHOD(last_reset_num_mesh_cells)
  MAKE_GETTER_METHOD(last_reset_max_num_collision_cells)
};

} // namespace

TEST(DSMC, collision_cell_rate_reduction) {

  int npart_cell = 511;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto sycl_target = sycl_target_t;
  const auto cell_count = cell_count_t;

  A->add_particle_dat(Sym<INT>("SPECIES_ID"), 1);
  A->add_particle_dat(Sym<INT>("MASK"), 1);
  A->add_particle_dat(Sym<INT>("COLLISION_CELL"), 1);

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng_state(52234234 + rank);
  std::uniform_real_distribution<> rng_dist(0.0, 1.0);

  auto rng_lambda = [&]() -> REAL { return rng_dist(rng_state); };

  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 2);

  auto aa = particle_sub_group(A, []() { return true; });

  const int num_species = 2;
  const int species_id_offset = 3;
  const int num_collision_cells = 7;

  particle_loop(
      A,
      [=](auto INDEX, auto SPECIES_ID, auto COLLISION_CELL, auto RNG,
          auto MASK) {
        SPECIES_ID.at(0) =
            RNG.at(INDEX, 0) < 0.8 ? species_id_offset : species_id_offset + 1;

        const int num_collision_cells_inner =
            (INDEX.cell % num_collision_cells) + 1;

        COLLISION_CELL.at(0) = INDEX.layer % num_collision_cells_inner;
        MASK.at(0) = 0.5 < RNG.at(INDEX, 1) ? 0 : 1;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("SPECIES_ID")),
      Access::write(Sym<INT>("COLLISION_CELL")), Access::read(rng_kernel),
      Access::write(Sym<INT>("MASK")))
      ->execute();

  std::vector<INT> species_ids(num_species);
  std::iota(species_ids.begin(), species_ids.end(), species_id_offset);

  std::shared_ptr<DSMC::CollisionCellPartition> collision_cell_partition =
      std::make_shared<DSMC::CollisionCellPartition>(sycl_target, cell_count,
                                                     species_ids);

  std::vector<int> collision_cell_counts(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    collision_cell_counts.at(cx) = (cx % num_collision_cells) + 1;
  }

  collision_cell_partition->construct(aa, collision_cell_counts,
                                      Sym<INT>("SPECIES_ID"), 0,
                                      Sym<INT>("COLLISION_CELL"), 0);

  auto collision_cell_rate_reduction =
      std::make_shared<TestCollisionCellRateReduction>(
          collision_cell_partition);

  ASSERT_EQ(collision_cell_rate_reduction->get_num_contributors(), -1);
  ASSERT_EQ(collision_cell_rate_reduction->get_num_entries(), -1);
  ASSERT_EQ(collision_cell_rate_reduction->get_last_reset_num_mesh_cells(), -1);
  ASSERT_EQ(
      collision_cell_rate_reduction->get_last_reset_max_num_collision_cells(),
      -1);

  const int num_contributors = 2;
  collision_cell_rate_reduction->setup(num_contributors);
  ASSERT_EQ(collision_cell_rate_reduction->get_num_contributors(), 2);

  collision_cell_rate_reduction->resize();
  ASSERT_EQ(collision_cell_rate_reduction->get_last_reset_num_mesh_cells(),
            cell_count);
  ASSERT_EQ(
      collision_cell_rate_reduction->get_last_reset_max_num_collision_cells(),
      num_collision_cells);

  std::vector<int> h_tmp_int(cell_count);
  sycl_target->queue
      .memcpy(h_tmp_int.data(),
              collision_cell_rate_reduction->get_d_num_collision_cells()->ptr,
              cell_count * sizeof(int))
      .wait_and_throw();

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(h_tmp_int.at(cx), (cx % num_collision_cells) + 1);
  }

  std::vector<REAL> h_tmp_real(num_contributors * cell_count *
                               num_collision_cells);
  auto d_accumulation_max =
      collision_cell_rate_reduction->get_d_accumulation_max();

  ASSERT_TRUE(d_accumulation_max->size >=
              static_cast<std::size_t>(num_contributors * cell_count *
                                       num_collision_cells));

  auto lambda_get_max = [&]() {
    sycl_target->queue
        .memcpy(h_tmp_real.data(), d_accumulation_max->ptr,
                num_contributors * cell_count * num_collision_cells *
                    sizeof(REAL))
        .wait_and_throw();
  };
  lambda_get_max();

  for (int contribx = 0; contribx < num_contributors; contribx++) {
    const std::size_t offset = contribx * cell_count * num_collision_cells;
    for (int mx = 0; mx < cell_count; mx++) {
      for (int cx = 0; cx < num_collision_cells; cx++) {
        ASSERT_EQ(h_tmp_real.at(offset + mx * num_collision_cells + cx), 0.0);
      }
    }
  }

  {
    NDLocalArraySharedPtr<REAL, 2> ndla_add = nullptr;
    collision_cell_rate_reduction->get(ndla_add);
    ASSERT_TRUE(ndla_add != nullptr);

    auto correct_shape = nd_index<2>(cell_count, num_collision_cells);
    ASSERT_EQ(correct_shape, ndla_add->index);

    NDHostArraySharedPtr<REAL, 2> h_ndla_add = nullptr;
    ndla_add->get(h_ndla_add);

    for (int mx = 0; mx < cell_count; mx++) {
      for (int cx = 0; cx < num_collision_cells; cx++) {
        ASSERT_EQ(h_ndla_add->at(mx, cx), 0.0);
      }
    }
  }

  std::vector<int> h_cell_mask(cell_count);
  std::fill(h_cell_mask.begin(), h_cell_mask.end(), 0);
  for (int mx = 0; mx < cell_count; mx++) {
    if (mx % 2 == 0) {
      h_cell_mask.at(mx) = 1;
    }
  }

  std::map<int, std::map<int, REAL>> map_contrib_to_rate;
  map_contrib_to_rate[0][0] = 0.0;
  map_contrib_to_rate[1][0] = 0.0;
  map_contrib_to_rate[0][1] = 0.0;
  map_contrib_to_rate[1][1] = 0.0;

  auto lambda_check_entries = [&]() {
    lambda_get_max();

    for (int contribx = 0; contribx < num_contributors; contribx++) {
      const std::size_t offset = contribx * cell_count * num_collision_cells;
      for (int mx = 0; mx < cell_count; mx++) {
        for (int cx = 0; cx < num_collision_cells; cx++) {
          const int keyx = mx % 2;
          const REAL set_correct = map_contrib_to_rate.at(contribx).at(keyx);
          const REAL to_test =
              h_tmp_real.at(offset + mx * num_collision_cells + cx);

          const REAL correct =
              (cx < collision_cell_counts.at(mx)) ? set_correct : 0.0;

          ASSERT_EQ(correct, to_test);
        }
      }
    }
  };

  collision_cell_rate_reduction->update(0, h_cell_mask, 1.1);
  map_contrib_to_rate[0][0] = 1.1;
  lambda_check_entries();

  collision_cell_rate_reduction->update(1, h_cell_mask, 2.1);
  map_contrib_to_rate[1][0] = 2.1;
  lambda_check_entries();

  std::fill(h_cell_mask.begin(), h_cell_mask.end(), 0);
  for (int mx = 0; mx < cell_count; mx++) {
    if (mx % 2 == 1) {
      h_cell_mask.at(mx) = 1;
    }
  }

  collision_cell_rate_reduction->update(0, h_cell_mask, 3.1);
  map_contrib_to_rate[0][1] = 3.1;
  lambda_check_entries();
  collision_cell_rate_reduction->update(1, h_cell_mask, 4.1);
  map_contrib_to_rate[1][1] = 4.1;
  lambda_check_entries();

  {
    NDLocalArraySharedPtr<REAL, 2> ndla_add = nullptr;
    collision_cell_rate_reduction->get(ndla_add);
    ASSERT_TRUE(ndla_add != nullptr);

    auto correct_shape = nd_index<2>(cell_count, num_collision_cells);
    ASSERT_EQ(correct_shape, ndla_add->index);

    NDHostArraySharedPtr<REAL, 2> h_ndla_add = nullptr;
    ndla_add->get(h_ndla_add);

    for (int mx = 0; mx < cell_count; mx++) {
      for (int cx = 0; cx < num_collision_cells; cx++) {
        const REAL correct = cx < collision_cell_counts.at(mx)
                                 ? mx % 2 == 0 ? 1.1 + 2.1 : 3.1 + 4.1
                                 : 0.0;

        const REAL to_test = h_ndla_add->at(mx, cx);
        const REAL err = relative_error(correct, to_test);
        ASSERT_TRUE(err < 1.0e-14);
      }
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}
