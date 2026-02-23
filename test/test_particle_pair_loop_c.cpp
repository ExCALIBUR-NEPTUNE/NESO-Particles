#include "include/test_neso_particles.hpp"

TEST(ParticlePairLoopBlock, local_array_read_write_add) {

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
    ASSERT_EQ(dh_to_test_int.h_buffer.ptr[ix], correct_int);
    ASSERT_NEAR(dh_to_test_real.h_buffer.ptr[ix], correct_real, 1.0e-15);
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

  auto la_test_add_int = std::make_shared<LocalArray<INT>>(sycl_target, 2);
  la_test_add_int->fill(0);
  auto la_test_add_real = std::make_shared<LocalArray<REAL>>(sycl_target, 2);
  la_test_add_real->fill(0.0);

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto LA_INT, auto LA_REAL, auto P_i, auto P_j,
          auto INDEX_i, auto INDEX_j, auto LA_ADD_INT, auto LA_ADD_REAL) {
        const INT index = INDEX.get_loop_linear_index();
        const INT cell = INDEX_i.at(0);
        const INT layer_i = INDEX_i.at(1);
        const INT layer_j = INDEX_j.at(1);

        LA_INT.at(index * 3 + 0) = cell;
        LA_INT.at(index * 3 + 1) = layer_i;
        LA_INT.at(index * 3 + 2) = layer_j;

        const REAL p0 = P_i.at(0) + P_j.at(0);
        const REAL p1 = P_i.at(1) + P_j.at(1);

        LA_REAL.at(2 * index + 0) = p0;
        LA_REAL.at(2 * index + 1) = p1;

        LA_ADD_INT.fetch_add(0, 1);
        LA_ADD_INT.fetch_add(1, -1);

        LA_ADD_REAL.fetch_add(0, 0.01 * p0);
        LA_ADD_REAL.fetch_add(1, -0.01 * p1);
      },
      Access::read(ParticlePairLoopIndex{}), Access::write(la_test_int),
      Access::write(la_test_real), Access::A(Access::read(Sym<REAL>("P"))),
      Access::B(Access::read(Sym<REAL>("P"))),
      Access::A(Access::read(Sym<INT>("INDEX"))),
      Access::B(Access::read(Sym<INT>("INDEX"))), Access::add(la_test_add_int),
      Access::add(la_test_add_real))
      ->execute();

  dh_to_test_int.device_to_host();
  dh_to_test_real.device_to_host();

  auto h_test_int = la_test_int->get();
  auto h_test_real = la_test_real->get();
  auto h_test_add_int = la_test_add_int->get();
  auto h_test_add_real = la_test_add_real->get();

  REAL correct_add_real0 = 0.0;
  REAL correct_add_real1 = 0.0;

  for (int ix = 0; ix < total_num_pairs; ix++) {
    const auto cell = static_cast<int>(h_test_int.at(3 * ix + 0));
    const auto layer_i = static_cast<int>(h_test_int.at(3 * ix + 1));
    const auto layer_j = static_cast<int>(h_test_int.at(3 * ix + 2));

    auto P = A->get_cell(Sym<REAL>("P"), cell);

    REAL P_c0 = P->at(layer_i, 0) + P->at(layer_j, 0);
    REAL P_c1 = P->at(layer_i, 1) + P->at(layer_j, 1);

    ASSERT_NEAR(h_test_real.at(2 * ix + 0), P_c0, 1.0e-14);
    ASSERT_NEAR(h_test_real.at(2 * ix + 1), P_c1, 1.0e-14);

    correct_add_real0 += 0.01 * P_c0;
    correct_add_real1 -= 0.01 * P_c1;
  }

  ASSERT_EQ(h_test_add_int.at(0), total_num_pairs);
  ASSERT_EQ(h_test_add_int.at(1), -total_num_pairs);

  ASSERT_NEAR(relative_error(correct_add_real0, h_test_add_real.at(0)), 0.0,
              1.0e-8);
  ASSERT_NEAR(relative_error(correct_add_real1, h_test_add_real.at(1)), 0.0,
              1.0e-8);

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticlePairLoopBlock, nd_local_array_read_write_add) {

  int npart_cell = 127;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("INDEX"), 2);

  particle_loop(
      A,
      [=](auto INDEX, auto INDEX_PARTICLE) {
        INDEX_PARTICLE.at(0) = INDEX.cell;
        INDEX_PARTICLE.at(1) = INDEX.layer;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("INDEX")))
      ->execute();

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
  const int total_num_pairs = cell_count * num_pairs_per_cell;
  std::fill(num_pairs.begin(), num_pairs.end(), num_pairs_per_cell);
  pair_sampler_ntc->sample(aa, aa, num_pairs);
  ASSERT_TRUE(pair_sampler_ntc->validate_pair_list(sycl_target));

  auto ndla_int = std::make_shared<NDLocalArray<int, 2>>(sycl_target, 3, 5);
  auto h_ndla_int = ndla_int->get();

  auto ndla_int_npair = std::make_shared<NDLocalArray<int, 3>>(
      sycl_target, 1, 1, total_num_pairs);
  ndla_int_npair->fill(0);

  int index = 0;
  int correct_sum = 0;
  for (int i0 = 0; i0 < 3; i0++) {
    for (int i1 = 0; i1 < 5; i1++) {
      h_ndla_int.at(index) = index;
      correct_sum += index;
      index++;
    }
  }
  ndla_int->set(h_ndla_int);

  auto ndla_real = std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, 3, 5);
  ndla_real->fill(0.0);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto INDEX, auto NDLA_INT, auto NDLA_REAL, auto NDLA_NPAIR) {
        int to_test_sum = 0;
        for (int i0 = 0; i0 < 3; i0++) {
          for (int i1 = 0; i1 < 5; i1++) {
            int index = NDLA_INT.at(i0, i1);
            to_test_sum += index;
            NDLA_REAL.fetch_add(i0, i1, 0.01 * (i0 + i1));
          }
        }
        NESO_KERNEL_ASSERT(to_test_sum == correct_sum, k_ep);
        const int index = static_cast<int>(INDEX.get_loop_linear_index());
        NDLA_NPAIR.at(0, 0, index) = 42 + index;
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(ndla_int),
      Access::add(ndla_real), Access::write(ndla_int_npair))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  auto h_ndla_real = ndla_real->get();
  index = 0;
  for (int i0 = 0; i0 < 3; i0++) {
    for (int i1 = 0; i1 < 5; i1++) {
      const REAL correct = total_num_pairs * 0.01 * (i0 + i1);
      const REAL to_test = h_ndla_real.at(index);
      ASSERT_NEAR(relative_error(correct, to_test), 0, 1.0e-8);
      index++;
    }
  }

  auto h_ndla_int_npair = ndla_int_npair->get();
  for (int ix = 0; ix < total_num_pairs; ix++) {
    ASSERT_EQ(h_ndla_int_npair.at(ix), ix + 42);
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticlePairLoopBlock, sym_vector) {

  int npart_cell = 127;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NUM_N"), 2);
  A->add_particle_dat(Sym<INT>("FOO"), 2);
  A->add_particle_dat(Sym<REAL>("R0"), 1);
  A->add_particle_dat(Sym<REAL>("R1"), 1);

  particle_loop(
      A,
      [=](auto INDEX, auto NN, auto FOO, auto R0, auto R1, auto V) {
        NN.at(0) = 0;
        NN.at(1) = 0;
        FOO.at(0) = 0;
        FOO.at(1) = 0;
        R0.at(0) = INDEX.get_loop_linear_index() * 0.01;
        R1.at(0) = INDEX.get_loop_linear_index() * 0.01;
        V.at(0) = 0.0;
        V.at(1) = 0.0;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("NUM_N")),
      Access::write(Sym<INT>("FOO")), Access::write(Sym<REAL>("R0")),
      Access::write(Sym<REAL>("R1")), Access::write(Sym<REAL>("V")))
      ->execute();

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

  const int num_pairs_per_cell = 221;
  std::vector<int> num_pairs(cell_count);
  std::fill(num_pairs.begin(), num_pairs.end(), num_pairs_per_cell);
  pair_sampler_ntc->sample(aa, aa, num_pairs);

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto NN_a, auto NN_b, auto SVI_a, auto SVI_b) {
        NN_a.at(0)++;
        NN_b.at(0)++;

        SVI_a.at(1, 1)++;
        SVI_b.at(1, 1)++;
      },
      Access::A(Access::write(Sym<INT>("NUM_N"))),
      Access::B(Access::write(Sym<INT>("NUM_N"))),
      Access::A(Access::write(
          sym_vector<INT>(A, {Sym<INT>("NUM_N"), Sym<INT>("FOO")}))),
      Access::B(Access::write(
          sym_vector<INT>(A, {Sym<INT>("NUM_N"), Sym<INT>("FOO")}))))
      ->execute();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      A,
      [=](auto NN, auto FOO) {
        NESO_KERNEL_ASSERT(NN.at(0) == FOO.at(1), k_ep);
      },
      Access::read(Sym<INT>("NUM_N")), Access::read(Sym<INT>("FOO")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
          A, A, pair_sampler_ntc)},
      [=](auto V_a, auto V_b, auto R0_a, auto R0_b, auto SR1_a, auto SR1_b) {
        V_a.at(0) += R0_b.at(0);
        V_a.at(1) += SR1_b.at(1, 0);

        V_b.at(0) += R0_a.at(0);
        V_b.at(1) += SR1_a.at(1, 0);
      },
      Access::A(Access::write(Sym<REAL>("V"))),
      Access::B(Access::write(Sym<REAL>("V"))),
      Access::A(Access::read(Sym<REAL>("R0"))),
      Access::B(Access::read(Sym<REAL>("R0"))),
      Access::A(
          Access::read(sym_vector<REAL>(A, {Sym<REAL>("P"), Sym<REAL>("R1")}))),
      Access::B(
          Access::read(sym_vector<REAL>(A, {Sym<REAL>("P"), Sym<REAL>("R1")}))))
      ->execute();

  particle_loop(
      A,
      [=](auto V) {
        NESO_KERNEL_ASSERT(relative_error(V.at(0), V.at(1)) < 1.0e-10, k_ep);
      },
      Access::read(Sym<REAL>("V")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
  A->domain->mesh->free();
}
