#include "include/test_neso_particles.hpp"

TEST(ParticlePairLoopBlock, local_array_read_write) {

  int npart_cell = 127;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("INDEX"), 2);

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

  const int num_pairs_per_cell = 17;
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

  la_test_int =
      std::make_shared<LocalArray<INT>>(sycl_target, 3 * total_num_pairs);
  la_test_real =
      std::make_shared<LocalArray<REAL>>(sycl_target, 2 * total_num_pairs);

  particle_loop(
      A,
      [=](auto INDEX, auto INDEX_PARTICLE) {
        INDEX_PARTICLE.at(0) = INDEX.cell;
        INDEX_PARTICLE.at(1) = INDEX.layer;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("INDEX")))
      ->execute();

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto LA_INT, auto LA_REAL, auto P_i, auto P_j,
          auto INDEX_i, auto INDEX_j) {
        const INT index = INDEX.get_loop_linear_index();
        const INT cell = INDEX_i.at(0);
        const INT layer_i = INDEX_i.at(1);
        const INT layer_j = INDEX_j.at(1);

        LA_INT.at(index * 3 + 0) = cell;
        LA_INT.at(index * 3 + 1) = layer_i;
        LA_INT.at(index * 3 + 2) = layer_j;

        for (int dx = 0; dx < 2; dx++) {
          LA_REAL.at(2 * index + dx) = P_i.at(dx) + P_j.at(dx);
        }
      },
      Access::read(ParticlePairLoopIndex{}), Access::write(la_test_int),
      Access::write(la_test_real), Access::A(Access::read(Sym<REAL>("P"))),
      Access::B(Access::read(Sym<REAL>("P"))),
      Access::A(Access::read(Sym<INT>("INDEX"))),
      Access::B(Access::read(Sym<INT>("INDEX"))))
      ->execute();

  dh_to_test_int.device_to_host();
  dh_to_test_real.device_to_host();

  auto h_test_int = la_test_int->get();
  auto h_test_real = la_test_real->get();

  for (int ix = 0; ix < total_num_pairs; ix++) {
    const auto cell = static_cast<int>(h_test_int.at(3 * ix + 0));
    const auto layer_i = static_cast<int>(h_test_int.at(3 * ix + 1));
    const auto layer_j = static_cast<int>(h_test_int.at(3 * ix + 2));

    auto P = A->get_cell(Sym<REAL>("P"), cell);

    REAL P_c0 = P->at(layer_i, 0) + P->at(layer_j, 0);
    REAL P_c1 = P->at(layer_i, 1) + P->at(layer_j, 1);

    ASSERT_NEAR(h_test_real.at(2 * ix + 0), P_c0, 1.0e-14);
    ASSERT_NEAR(h_test_real.at(2 * ix + 1), P_c1, 1.0e-14);
  }

  sycl_target->free();
  A->domain->mesh->free();
}
