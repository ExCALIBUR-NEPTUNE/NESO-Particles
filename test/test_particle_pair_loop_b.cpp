#include "include/test_neso_particles.hpp"

TEST(ParticlePairLoopBlock, base) {

  int npart_cell = 10;
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
      [](auto NN_i, auto NN_j) {
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
