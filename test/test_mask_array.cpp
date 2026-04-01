#include "include/test_neso_particles.hpp"

TEST(MaskArray, base) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  MaskArrayDevice h_mad = {nullptr, 2, 14};

  for (MaskArrayBaseType ix = 0; ix < h_mad.num_bits_per_base; ix++) {
    ASSERT_EQ(h_mad.get_inner_index(7 + ix, 0),
              (7 + ix) % h_mad.num_bits_per_base);
    ASSERT_EQ(h_mad.get_outer_index(7 + ix, 0),
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

      const std::size_t stride = ma->get_stride(N);

      ASSERT_EQ(ma->size, 0);
      ma->reset(true, N);
      ASSERT_EQ(ma->size, N);

      MaskArrayDevice d_ma = ma->get_device();

      ASSERT_EQ(d_ma.num_masks_per_entry, B);
      ASSERT_EQ(d_ma.stride, ma->stride);
      ASSERT_EQ(stride, ma->stride);

      const std::size_t M = stride * ma->num_masks_per_entry;

      ASSERT_TRUE(M * ma->num_bits_per_base >= N * B);

      h_ma.clear();
      h_ma.resize(M);
      std::fill(h_ma.begin(), h_ma.end(), 0);
      MaskArrayDevice h_mad = {h_ma.data(), d_ma.num_masks_per_entry, stride};

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
