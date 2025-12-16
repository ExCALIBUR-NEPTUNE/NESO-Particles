#include "include/test_neso_particles.hpp"

TEST(ParticlePairLoopBlock, base) {

  int npart_cell = 127;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NEIGHBOURS"), 2);

  auto reset_loop = particle_loop(
      A,
      [=](auto NN) {
        NN.at(0) = 0;
        NN.at(1) = 1;
      },
      Access::write(Sym<INT>("NEIGHBOURS")));

  reset_loop->execute();
  auto aa = particle_sub_group(A, []() { return true; });

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
  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto NN_i, auto NN_j) {
        NN_i.at(0)++;
        NN_j.at(0)++;
      },
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))))
      ->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cellx);
    const int nrow = NN->nrow;
    for (int rx = 0; rx < nrow; rx++) {
      const INT to_test = NN->at(rx, 0);
      ASSERT_EQ(0, to_test);
    }
  }

  std::fill(num_pairs.begin(), num_pairs.end(), 127);
  pair_sampler_ntc->sample(aa, aa, num_pairs);
  ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [](auto NN_i, auto NN_j) {
        NN_i.at(0)++;
        NN_j.at(0)++;
      },
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))))
      ->execute();

  std::map<std::pair<int, int>, int> map_particles_to_nn;

  auto h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int num_pairs_test =
        static_cast<int>(std::get<0>(h_pair_list[cellx]).size());
    ASSERT_EQ(num_pairs_test, num_pairs.at(cellx));
    for (int ix = 0; ix < num_pairs_test; ix++) {
      const int i = std::get<0>(h_pair_list[cellx])[ix];
      const int j = std::get<1>(h_pair_list[cellx])[ix];
      map_particles_to_nn[{cellx, i}]++;
      map_particles_to_nn[{cellx, j}]++;
    }
  }

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cellx);
    const int nrow = NN->nrow;

    for (int rx = 0; rx < nrow; rx++) {
      const INT to_test = NN->at(rx, 0);
      const INT correct = map_particles_to_nn[{cellx, rx}];
      ASSERT_EQ(correct, to_test);
    }
  }

  int total_num_pairs = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    num_pairs[cellx] = cellx;
    total_num_pairs += cellx;
  }

  pair_sampler_ntc->sample(aa, aa, num_pairs);
  ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));

  std::vector<int> h_flags(total_num_pairs);
  std::fill(h_flags.begin(), h_flags.end(), 0);

  BufferDevice<int> d_flags(sycl_target, h_flags);
  int *k_flags = d_flags.ptr;

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX) {
        const int linear_index = INDEX.get_loop_linear_index();
        NESO_KERNEL_ASSERT(k_flags[linear_index] == 0, k_ep);
        k_flags[linear_index] = linear_index;
      },
      Access::read(ParticlePairLoopIndex{}))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  h_flags = d_flags.get();
  for (int ix = 0; ix < total_num_pairs; ix++) {
    ASSERT_EQ(h_flags.at(ix), ix);
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticlePairLoopBlock, kernel_rng_base) {

  int npart_cell = 255;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NEIGHBOURS"), 2);

  auto reset_loop = particle_loop(
      A,
      [=](auto NN) {
        NN.at(0) = 0;
        NN.at(1) = 1;
      },
      Access::write(Sym<INT>("NEIGHBOURS")));

  reset_loop->execute();
  auto aa = particle_sub_group(A, []() { return true; });

  const int rank = sycl_target->comm_pair.rank_parent;

  std::mt19937 rng(34234 + rank);
  std::uniform_real_distribution<REAL> dist{
      std::uniform_real_distribution<REAL>(0.0, 1.0)};
  auto lambda_sampler = [&]() -> REAL { return dist(rng); };

  auto rng_function_ntc =
      std::make_shared<HostRNGGenerationFunction<REAL>>(lambda_sampler);

  auto lambda_sampler_kernel = [&]() -> REAL { return dist(rng) + 1.0; };

  auto rng_function_kernel =
      std::make_shared<HostRNGGenerationFunction<REAL>>(lambda_sampler_kernel);

  auto pair_sampler_ntc = std::make_shared<DSMC::PairSamplerNTC>(
      sycl_target, cell_count, rng_function_ntc);

  std::vector<int> num_pairs(cell_count);
  std::fill(num_pairs.begin(), num_pairs.end(), 0);
  pair_sampler_ntc->sample(aa, aa, num_pairs);

  const int rng_ncomp = 2;
  auto rng_block_kernel = host_per_particle_block_rng<REAL>(
      std::dynamic_pointer_cast<RNGGenerationFunction<REAL>>(
          rng_function_kernel),
      rng_ncomp);

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto RNG, auto P2_i, auto P2_j) {
        bool valid = true;
        P2_i.at(0) = RNG.at(INDEX, 0, &valid);
        P2_j.at(0) = RNG.at(INDEX, 1, &valid);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(rng_block_kernel),
      Access::A(Access::write(Sym<REAL>("P2"))),
      Access::B(Access::write(Sym<REAL>("P2"))))
      ->execute();

  ASSERT_EQ(static_cast<std::size_t>(0),
            rng_function_kernel->get_last_sample_size());

  auto rng_atomic_kernel = host_atomic_block_kernel_rng<REAL>(
      std::dynamic_pointer_cast<RNGGenerationFunction<REAL>>(
          rng_function_kernel),
      rng_ncomp);

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto RNG, auto P2_i, auto P2_j) {
        bool valid = true;
        P2_i.at(0) = RNG.at(INDEX, 0, &valid);
        P2_j.at(0) = RNG.at(INDEX, 1, &valid);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(rng_atomic_kernel),
      Access::A(Access::write(Sym<REAL>("P2"))),
      Access::B(Access::write(Sym<REAL>("P2"))))
      ->execute();

  ASSERT_EQ(static_cast<std::size_t>(0),
            rng_function_kernel->get_last_sample_size());

  int total_num_pairs = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    num_pairs.at(cellx) = cellx % 127;
    total_num_pairs += cellx % 127;
  }

  pair_sampler_ntc->sample(aa, aa, num_pairs);

  auto reset_P2 = particle_loop(
      A, [=](auto P2) { P2.at(0) = -1.0; }, Access::write(Sym<REAL>("P2")));

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto RNG, auto P2_i, auto P2_j) {
        bool valid = true;
        P2_i.at(0) = RNG.at(INDEX, 0, &valid);
        NESO_KERNEL_ASSERT(valid, k_ep);
        P2_j.at(0) = RNG.at(INDEX, 1, &valid);
        NESO_KERNEL_ASSERT(valid, k_ep);
        NESO_KERNEL_ASSERT(P2_i.at(0) >= 1.0, k_ep);
        NESO_KERNEL_ASSERT(P2_i.at(0) <= 2.0, k_ep);
        NESO_KERNEL_ASSERT(P2_j.at(0) >= 1.0, k_ep);
        NESO_KERNEL_ASSERT(P2_j.at(0) <= 2.0, k_ep);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(rng_block_kernel),
      Access::A(Access::write(Sym<REAL>("P2"))),
      Access::B(Access::write(Sym<REAL>("P2"))))
      ->execute();

  ASSERT_EQ(static_cast<std::size_t>(total_num_pairs * rng_ncomp),
            rng_function_kernel->get_last_sample_size());
  ASSERT_FALSE(ep.get_flag());

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto RNG, auto P2_i, auto P2_j) {
        bool valid = true;
        P2_i.at(0) = RNG.at(INDEX, 0, &valid);
        NESO_KERNEL_ASSERT(valid, k_ep);
        P2_j.at(0) = RNG.at(INDEX, 1, &valid);
        NESO_KERNEL_ASSERT(valid, k_ep);
        NESO_KERNEL_ASSERT(P2_i.at(0) >= 1.0, k_ep);
        NESO_KERNEL_ASSERT(P2_i.at(0) <= 2.0, k_ep);
        NESO_KERNEL_ASSERT(P2_j.at(0) >= 1.0, k_ep);
        NESO_KERNEL_ASSERT(P2_j.at(0) <= 2.0, k_ep);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(rng_atomic_kernel),
      Access::A(Access::write(Sym<REAL>("P2"))),
      Access::B(Access::write(Sym<REAL>("P2"))))
      ->execute();

  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());
  ASSERT_EQ(static_cast<std::size_t>(total_num_pairs * rng_ncomp),
            rng_function_kernel->get_last_sample_size());
  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticlePairLoopBlock, multiple_lists) {

  int npart_cell = 129;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;
  const int nbins = 3;
  const int nsteps = 4;
  const int npairs = 8;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NEIGHBOURS"), 2);
  A->add_particle_dat(Sym<INT>("DSMC_CELL"), 1);

  // RNG
  const int rank = sycl_target->comm_pair.rank_parent;
  std::mt19937 rng(34234 + rank);
  std::uniform_real_distribution<REAL> dist{
      std::uniform_real_distribution<REAL>(0.0, 1.0)};

  // NTC Sampler
  auto lambda_sampler = [&]() -> REAL { return dist(rng); };
  auto rng_function_ntc =
      std::make_shared<HostRNGGenerationFunction<REAL>>(lambda_sampler);

  auto reset_loop = particle_loop(
      A,
      [=](auto NN) {
        NN.at(0) = 0;
        NN.at(1) = 1;
      },
      Access::write(Sym<INT>("NEIGHBOURS")));
  reset_loop->execute();

  particle_loop(
      A, [=](auto ID, auto DSMC_CELL) { DSMC_CELL.at(0) = ID.at(0) % nbins; },
      Access::read(Sym<INT>("ID")), Access::write(Sym<INT>("DSMC_CELL")))
      ->execute();

  std::vector<ParticleSubGroupSharedPtr> sub_groups =
      particle_group_partition(A, Sym<INT>("DSMC_CELL"), nbins);

  std::vector<DSMC::PairSamplerNTCSharedPtr> ntc_pair_lists(nbins);
  std::vector<
      CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>>
      ntc_abs_pair_lists(nbins);

  for (int binx = 0; binx < nbins; binx++) {
    ntc_pair_lists.at(binx) = std::make_shared<DSMC::PairSamplerNTC>(
        sycl_target, cell_count, rng_function_ntc);
    ntc_abs_pair_lists.at(binx) = {A, A, ntc_pair_lists.at(binx)};
  }

  std::vector<int> num_pairs(cell_count);
  std::fill(num_pairs.begin(), num_pairs.end(), 0);

  // Kernel RNG sampler
  auto lambda_sampler_kernel = [&]() -> REAL { return dist(rng) + 1.0; };
  auto rng_function_kernel =
      std::make_shared<HostRNGGenerationFunction<REAL>>(lambda_sampler_kernel);

  const int rng_ncomp = 1;
  auto rng_block_kernel = host_per_particle_block_rng<REAL>(
      std::dynamic_pointer_cast<RNGGenerationFunction<REAL>>(
          rng_function_kernel),
      rng_ncomp);

  auto d_test_linear_index = std::make_shared<BufferDevice<int>>(
      sycl_target, cell_count * nbins * npairs);
  auto k_test_linear_index = d_test_linear_index->ptr;
  auto h_test_linear_index = d_test_linear_index->get();
  std::fill(h_test_linear_index.begin(), h_test_linear_index.end(), 0);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  auto pl0 = particle_pair_loop(
      ntc_abs_pair_lists,
      [=](auto INDEX, auto DSMC_CELL_I, auto DSMC_CELL_J, auto RNG_KERNEL) {
        NESO_KERNEL_ASSERT(DSMC_CELL_I.at(0) == DSMC_CELL_J.at(0), k_ep);
        const auto linear_index = INDEX.get_loop_linear_index();
        k_test_linear_index[linear_index] = linear_index;
        bool valid = true;
        const REAL rng_sample = RNG_KERNEL.at(INDEX, 0, &valid);
        NESO_KERNEL_ASSERT((1.0 <= rng_sample) && (rng_sample <= 2.0), k_ep);
        NESO_KERNEL_ASSERT(valid, k_ep);
      },
      Access::read(ParticlePairLoopIndex{}),
      Access::A(Access::read(Sym<INT>("DSMC_CELL"))),
      Access::B(Access::read(Sym<INT>("DSMC_CELL"))),
      Access::read(rng_block_kernel));

  auto pl1 = particle_pair_loop(
      ntc_abs_pair_lists,
      [=](auto NN_i, auto NN_j) {
        NN_i.at(0)++;
        NN_j.at(0)++;
      },
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  auto permute_loop = particle_loop(
      A,
      [=](auto INDEX, auto DSMC_CELL, auto KERNEL_RNG) {
        bool valid = true;
        const int offset = (KERNEL_RNG.at(INDEX, 0, &valid) * nbins);
        const int new_dsmc_cell = (DSMC_CELL.at(0) + offset) % nbins;
        DSMC_CELL.at(0) = new_dsmc_cell;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("DSMC_CELL")),
      Access::read(rng_block_kernel));

  for (int stepx = 0; stepx < nsteps; stepx++) {
    reset_loop->execute();

    int total_num_pairs = 0;
    for (int binx = 0; binx < nbins; binx++) {
      auto groupx = sub_groups.at(binx);
      for (int cellx = 0; cellx < cell_count; cellx++) {
        const int npart_cell = groupx->get_npart_cell(cellx);
        const int npairs_cell = npart_cell > 1 ? npairs : 0;
        num_pairs.at(cellx) = npairs_cell;
        total_num_pairs += npairs_cell;
      }
      ntc_pair_lists.at(binx)->sample(groupx, groupx, num_pairs);
    }
    for (int binx = 0; binx < nbins; binx++) {
      ASSERT_TRUE(ntc_pair_lists.at(binx)->validate_pair_list(sycl_target));
    }

    d_test_linear_index->set(h_test_linear_index);
    pl0->execute();
    ASSERT_FALSE(ep.get_flag());

    auto h_test_linear_index2 = d_test_linear_index->get();
    for (int ix = 0; ix < total_num_pairs; ix++) {
      ASSERT_EQ(h_test_linear_index2.at(ix), ix);
    }

    ASSERT_EQ(total_num_pairs, rng_function_kernel->get_last_sample_size());

    pl1->execute();

    std::map<std::pair<int, int>, int> h_nn;

    for (int binx = 0; binx < nbins; binx++) {
      auto h_pair_list =
          ntc_pair_lists.at(binx)->get_host_pair_list(sycl_target);
      for (int cellx = 0; cellx < cell_count; cellx++) {
        auto &pair_i = std::get<0>(h_pair_list.at(cellx));
        auto &pair_j = std::get<1>(h_pair_list.at(cellx));
        const int tmp_num_pairs = static_cast<int>(pair_i.size());
        for (int px = 0; px < tmp_num_pairs; px++) {
          const int ix = pair_i.at(px);
          const int jx = pair_j.at(px);
          h_nn[{cellx, ix}]++;
          h_nn[{cellx, jx}]++;
        }
      }
    }

    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cellx);
      const int nrow = NN->nrow;
      for (int rowx = 0; rowx < nrow; rowx++) {
        const int nn_to_test = NN->at(rowx, 0);
        const int nn_correct = h_nn[{cellx, rowx}];
        ASSERT_EQ(nn_correct, nn_to_test);
      }
    }

    permute_loop->execute();
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticlePairLoopBlock, wave_analysis) {

  {
    std::vector<int> test_occupancy(33);
    std::fill(test_occupancy.begin(), test_occupancy.end(), 0);
    ASSERT_EQ(get_mean_wave_occupancy(test_occupancy), 0.0);
    test_occupancy.at(32) = 1;
    ASSERT_NEAR(get_mean_wave_occupancy(test_occupancy), 1.0, 1.0e-12);
    test_occupancy.at(32) = 2;
    ASSERT_NEAR(get_mean_wave_occupancy(test_occupancy), 1.0, 1.0e-12);
    test_occupancy.at(0) = 1;
    test_occupancy.at(32) = 0;
    ASSERT_NEAR(get_mean_wave_occupancy(test_occupancy), 0.0, 1.0e-12);
    test_occupancy.at(0) = 1;
    test_occupancy.at(16) = 1;
    ASSERT_NEAR(get_mean_wave_occupancy(test_occupancy), 0.25, 1.0e-12);
  }

  const int max_num_pairs_to_sample = 1001;
  int npart_cell = 1000;
  const int ndim = 2;
  const int nx = 32;
  const int ny = 16;
  const int nz = 48;

  auto [A, sycl_target_t, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto sycl_target = sycl_target_t;

  auto aa = particle_sub_group(A, [=]() { return true; });
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
  std::fill(num_pairs.begin(), num_pairs.end(), max_num_pairs_to_sample);

  pair_sampler_ntc->sample(aa, aa, num_pairs);

  auto d_pair_list = pair_sampler_ntc->get_pair_list();
  auto h_pair_list = pair_sampler_ntc->get_host_pair_list(sycl_target);

  ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));

  std::vector<int> wave_occupancies;
  pair_sampler_ntc->get_wave_occupancy_counts(sycl_target, wave_occupancies);

  // The NTC samplers should never emit empty waves
  ASSERT_EQ(wave_occupancies.at(0), 0);

  ASSERT_EQ(static_cast<std::size_t>(d_pair_list.block_size + 1),
            wave_occupancies.size());

  int num_entries = 0;
  for (int ix = 0; ix < (d_pair_list.block_size + 1); ix++) {
    num_entries += ix * wave_occupancies.at(ix);
  }

  ASSERT_EQ(d_pair_list.pair_count, max_num_pairs_to_sample * cell_count);
  ASSERT_EQ(num_entries, d_pair_list.pair_count);

  const REAL mean_occupancy = get_mean_wave_occupancy(wave_occupancies);
  ASSERT_TRUE(0.0 <= mean_occupancy);
  ASSERT_TRUE(mean_occupancy <= 1.0);

  std::vector<int> global_wave_occupancies;
  get_global_wave_occupancy_counts(sycl_target, wave_occupancies,
                                   global_wave_occupancies);

  for (int ix = 0; ix < (d_pair_list.block_size + 1); ix++) {
    const int local_count = wave_occupancies.at(ix);
    int global_count = 0;

    MPICHK(MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0,
                      sycl_target->comm_pair.comm_parent));

    if (sycl_target->comm_pair.rank_parent == 0) {
      ASSERT_EQ(global_count, global_wave_occupancies.at(ix));
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}
