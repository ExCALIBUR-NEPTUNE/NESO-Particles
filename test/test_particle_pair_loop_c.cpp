#include "include/test_neso_particles.hpp"

TEST(ParticlePairLoopBlock, local_array) {

  int npart_cell = 127;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);

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

  const int num_pairs_per_cell = 127;
  std::fill(num_pairs.begin(), num_pairs.end(), num_pairs_per_cell);
  pair_sampler_ntc->sample(aa, aa, num_pairs);
  ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));

  std::vector<INT> v_test_int = {42, 107};
  std::vector<REAL> v_test_real = {3.14, 2.75};

  auto la_test_int = std::make_shared<LocalArray<INT>>(sycl_target, v_test_int);
  auto la_test_real =
      std::make_shared<LocalArray<REAL>>(sycl_target, v_test_real);

  const int total_num_pairs = cell_count * num_pairs_per_cell;

  BufferDeviceHost<INT> dh_to_test_int(sycl_target, total_num_pairs);
  BufferDeviceHost<REAL> dh_to_test_real(sycl_target, total_num_pairs);

  auto k_to_test_int = dh_to_test_int.d_buffer.ptr;
  auto k_to_test_real = dh_to_test_real.d_buffer.ptr;

  sycl_target->queue
      .parallel_for(sycl::range<1>(total_num_pairs),
                    [=](auto ix) {
                      k_to_test_int[ix] = 0;
                      k_to_test_real[ix] = 0;
                    })
      .wait_and_throw();

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto LA_INT, auto LA_REAL) {
        const INT index = INDEX.get_loop_linear_index();
        k_to_test_int[index] = LA_INT.at(0) + LA_INT.at(1);
        k_to_test_real[index] = LA_REAL.at(0) + LA_REAL.at(1);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(la_test_int),
      Access::read(la_test_real))
      ->execute();

  dh_to_test_int.device_to_host();
  dh_to_test_real.device_to_host();

  const INT correct_int = 42 + 107;
  const REAL correct_real = 3.14 + 2.75;

  for (int ix = 0; ix < total_num_pairs; ix++) {
    ASSERT_EQ(dh_to_test_int.d_buffer.ptr[ix], correct_int);
    ASSERT_NEAR(dh_to_test_real.d_buffer.ptr[ix], correct_real, 1.0e-15);
  }

  sycl_target->free();
  A->domain->mesh->free();
}
