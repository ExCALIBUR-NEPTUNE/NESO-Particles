#include "include/test_neso_particles.hpp"

TEST(MaskArray, base) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  MaskArrayDevice h_mad = {nullptr, 2, 14};

  for (MaskArrayBaseType ix = 0; ix < h_mad.num_bits_per_base; ix++) {
    ASSERT_EQ(h_mad.get_inner_index(0, 7 + ix),
              (7 + ix) % h_mad.num_bits_per_base);
    ASSERT_EQ(h_mad.get_outer_index(0, 7 + ix),
              (7 + ix) / h_mad.num_bits_per_base);
  }

  MaskArrayBaseType b = 0;
  for (MaskArrayBaseType ix = 0; ix < h_mad.num_bits_per_base; ix++) {
    h_mad.set_inner(&b, ix, true);
    ASSERT_EQ(sycl::popcount(b), ix + 1);
    ASSERT_EQ(h_mad.get_inner(&b, ix), true);
    for (MaskArrayBaseType jx = 0; jx < ix; jx++) {
      ASSERT_EQ(h_mad.get_inner(&b, jx), true);
    }
    for (MaskArrayBaseType jx = ix + 1; jx < h_mad.num_bits_per_base; jx++) {
      ASSERT_EQ(h_mad.get_inner(&b, jx), false);
    }
  }
  for (MaskArrayBaseType ix = 0; ix < h_mad.num_bits_per_base; ix++) {
    h_mad.set_inner(&b, ix, false);
    ASSERT_EQ(h_mad.get_inner(&b, ix), false);
    ASSERT_EQ(sycl::popcount(b), h_mad.num_bits_per_base - ix - 1);
    for (MaskArrayBaseType jx = 0; jx < ix; jx++) {
      ASSERT_EQ(h_mad.get_inner(&b, jx), false);
    }
    for (MaskArrayBaseType jx = ix + 1; jx < h_mad.num_bits_per_base; jx++) {
      ASSERT_EQ(h_mad.get_inner(&b, jx), true);
    }
  }

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  sycl_target->queue.single_task([=]() {
    MaskArrayBaseType b = 0;
    for (MaskArrayBaseType ix = 0; ix < h_mad.num_bits_per_base; ix++) {
      h_mad.set_inner(&b, ix, true);
      NESO_KERNEL_ASSERT(sycl::popcount(b) == ix + 1, k_ep);
    }
    for (MaskArrayBaseType ix = 0; ix < h_mad.num_bits_per_base; ix++) {
      h_mad.set_inner(&b, ix, false);
      NESO_KERNEL_ASSERT(sycl::popcount(b) == h_mad.num_bits_per_base - ix - 1,
                         k_ep);
    }
  });

  ASSERT_FALSE(ep.get_flag());

  std::vector<MaskArrayBaseType> h_ma;
  for (std::size_t B : {0, 1, 2}) {
    for (std::size_t N : {0, 1, 4, 31, 61, 32}) {
      auto ma = std::make_shared<MaskArray>(sycl_target, B);

      auto lambda_test = [&](const bool value) {
        const std::uint32_t correct_popcount =
            value ? sizeof(MaskArrayBaseType) * CHAR_BIT : 0;
        const MaskArrayBaseType reset_value = ma->get_reset_mask(value);
        NESOASSERT(sycl::popcount(reset_value) == correct_popcount,
                   "Failed to compute correct reset value.");
      };

      lambda_test(true);
      lambda_test(false);

      ASSERT_EQ(ma->size, 0);
      ma->reset(true, N);
      ASSERT_EQ(ma->size, N);

      const std::size_t size = ma->size;
      MaskArrayDevice d_ma = ma->get_device();

      ASSERT_EQ(d_ma.num_masks_per_entry, B);
      ASSERT_EQ(d_ma.size, ma->size);
      ASSERT_EQ(size, ma->size);

      const std::size_t M = size * ma->num_base_elements_per_entry;

      ASSERT_TRUE(M * ma->num_bits_per_base >= N * B);

      h_ma.clear();
      h_ma.resize(M);
      std::fill(h_ma.begin(), h_ma.end(), 0);
      MaskArrayDevice h_mad = {h_ma.data(), d_ma.num_masks_per_entry, size};

      std::deque<std::pair<std::size_t, std::size_t>> a;
      std::deque<std::pair<std::size_t, std::size_t>> b;

      for (std::size_t ix = 0; ix < N; ix++) {
        for (std::size_t jx = 0; jx < B; jx++) {
          a.push_back({ix, jx});
        }
      }

      while (!a.empty()) {
        auto [ix, bx] = a.back();
        h_mad.set(ix, bx, true);
        b.push_back({ix, bx});
        a.pop_back();

        for (auto &jx : b) {
          ASSERT_TRUE(h_mad.get(jx.first, jx.second));
        }

        for (auto &jx : a) {
          ASSERT_FALSE(h_mad.get(jx.first, jx.second));
        }
      }

      int count = 0;
      for (auto ix : h_ma) {
        count += sycl::popcount(ix);
      }
      ASSERT_EQ(count, static_cast<int>(N * B));

      ma->reset(false, N);
      d_ma = ma->get_device();
      sycl_target->queue
          .memcpy(h_ma.data(), d_ma.d_masks, M * sizeof(MaskArrayBaseType))
          .wait_and_throw();
      for (std::size_t ix = 0; ix < M; ix++) {
        ASSERT_EQ(h_ma.at(ix), static_cast<MaskArrayBaseType>(0));
      }

      sycl_target->queue
          .single_task([=]() {
            for (std::size_t ix = 0; ix < (N * B); ix++) {

              const std::size_t ex = ix / B;
              const std::size_t bx = ix % B;
              d_ma.set(ex, bx, true);

              for (std::size_t jx = 0; jx <= ix; jx++) {
                const std::size_t ex = jx / B;
                const std::size_t bx = jx % B;
                NESO_KERNEL_ASSERT(d_ma.get(ex, bx), k_ep);
              }

              for (std::size_t jx = ix + 1; jx < (N * B); jx++) {
                const std::size_t ex = jx / B;
                const std::size_t bx = jx % B;
                NESO_KERNEL_ASSERT(!d_ma.get(ex, bx), k_ep);
              }
            }
          })
          .wait_and_throw();

      ASSERT_FALSE(ep.get_flag());

      std::fill(h_ma.begin(), h_ma.end(), 0);
      sycl_target->queue
          .memcpy(h_ma.data(), d_ma.d_masks, M * sizeof(MaskArrayBaseType))
          .wait_and_throw();

      count = 0;
      for (auto ix : h_ma) {
        count += sycl::popcount(ix);
      }
      ASSERT_EQ(count, static_cast<int>(N * B));
    }
  }
  sycl_target->free();
}

TEST(MaskArray, particle_loop) {
  int npart_cell = 51;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 20;
  const int nz = 48;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);

  auto A = A_t;
  auto sycl_target = sycl_target_t;

  for (std::size_t B : {1, 2, 3, 7, 33, 65}) {

    auto ma = std::make_shared<MaskArray>(sycl_target, B);
    ma->reset(false, A->get_npart_local());

    particle_loop(
        A,
        [=](auto INDEX, auto MA) {
          for (std::size_t bx = 0; bx < B; bx++) {
            const bool value = ((INDEX.cell + INDEX.layer) % (bx + 1)) == 0;
            MA.set(INDEX.get_loop_linear_index(), bx, value);
          }
        },
        Access::read(ParticleLoopIndex{}), Access::write(ma))
        ->execute();

    ErrorPropagate ep(sycl_target);
    auto k_ep = ep.device_ptr();

    particle_loop(
        A,
        [=](auto INDEX, auto MA) {
          for (std::size_t bx = 0; bx < B; bx++) {
            const bool correct = ((INDEX.cell + INDEX.layer) % (bx + 1)) == 0;
            const bool to_test = MA.get(INDEX.get_loop_linear_index(), bx);
            NESO_KERNEL_ASSERT(correct == to_test, k_ep);
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(ma))
        ->execute();

    ASSERT_FALSE(ep.get_flag());
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(MaskArray, particle_pair_loop) {

  const int npart_cell = 257;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  auto cell_count = cell_count_t;

  const int num_samples = cell_count * npart_cell * 0.2;
  std::vector<int> h_c(num_samples);
  std::vector<int> h_i(num_samples);
  std::vector<int> h_j(num_samples);

  std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
  std::uniform_int_distribution<int> dist_cell(0, cell_count - 1);

  for (int ix = 0; ix < num_samples; ix++) {
    const int cell = dist_cell(rng);
    h_c[ix] = cell;
    const int npart_cell = A->get_npart_cell(cell);
    std::uniform_int_distribution<int> dist_layer(0, npart_cell - 1);
    h_i[ix] = dist_layer(rng);
    h_j[ix] = dist_layer(rng);
  }

  auto cellwise_pair_list =
      std::make_shared<CellwisePairListSimple>(sycl_target, cell_count);
  cellwise_pair_list->push_back(h_c, h_i, h_j);

  auto ma = std::make_shared<MaskArray>(sycl_target, 1);
  ma->reset(false, cellwise_pair_list->get_num_pairs());

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_list)},
      [=](auto INDEX, auto MASK_ARRAY) {
        NESO_KERNEL_ASSERT(
            MASK_ARRAY.get(INDEX.get_loop_linear_index(), 0) == false, k_ep);
        MASK_ARRAY.set(INDEX.get_loop_linear_index(), 0, true);
      },
      Access::read(ParticlePairLoopIndex{}), Access::write(ma))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_list)},
      [=](auto INDEX, auto MASK_ARRAY) {
        NESO_KERNEL_ASSERT(MASK_ARRAY.get(INDEX.get_loop_linear_index(), 0),
                           k_ep);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(ma))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  const INT num_set = ma->get_num_masks_true(0);
  ASSERT_EQ(num_set, cellwise_pair_list->get_num_pairs());

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(PairMask, particle_pair_loop) {

  const int npart_cell = 257;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  auto cell_count = cell_count_t;

  const int num_samples = cell_count * npart_cell * 0.2;
  std::vector<int> h_c(num_samples);
  std::vector<int> h_i(num_samples);
  std::vector<int> h_j(num_samples);

  std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
  std::uniform_int_distribution<int> dist_cell(0, cell_count - 1);

  for (int ix = 0; ix < num_samples; ix++) {
    const int cell = dist_cell(rng);
    h_c[ix] = cell;
    const int npart_cell = A->get_npart_cell(cell);
    std::uniform_int_distribution<int> dist_layer(0, npart_cell - 1);
    h_i[ix] = dist_layer(rng);
    h_j[ix] = dist_layer(rng);
  }

  auto cellwise_pair_list =
      std::make_shared<CellwisePairListSimple>(sycl_target, cell_count);
  cellwise_pair_list->push_back(h_c, h_i, h_j);

  auto ma = std::make_shared<PairMask>(sycl_target);
  ma->reset(false, cellwise_pair_list->get_num_pairs());

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_list)},
      [=](auto MASK_ARRAY) {
        NESO_KERNEL_ASSERT(MASK_ARRAY.get() == false, k_ep);
        MASK_ARRAY.set_on();
        NESO_KERNEL_ASSERT(MASK_ARRAY.get() == true, k_ep);
        MASK_ARRAY.set_off();
        NESO_KERNEL_ASSERT(MASK_ARRAY.get() == false, k_ep);
        MASK_ARRAY.set_on();
        NESO_KERNEL_ASSERT(MASK_ARRAY.get() == true, k_ep);
      },
      Access::write(ma))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_list)},
      [=](auto MASK_ARRAY) {
        NESO_KERNEL_ASSERT(MASK_ARRAY.get() == true, k_ep);
      },
      Access::write(ma))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  const INT num_set = ma->get_num_masks_true(0);
  ASSERT_EQ(num_set, cellwise_pair_list->get_num_pairs());

  sycl_target->free();
  A->domain->mesh->free();
}
