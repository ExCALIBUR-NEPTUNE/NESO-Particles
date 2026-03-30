#include "include/test_neso_particles.hpp"

TEST(MaskArray, base) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::vector<MaskArrayDevice> h_ma;

  for (std::size_t B : {0, 1, 2}) {
    for (std::size_t N : {0, 1, 31, 61, 32}) {
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

      const MaskArrayDevice d_ma = ma->get_device();

      ASSERT_EQ(d_ma.num_masks_per_entry, B);
      ASSERT_EQ(d_ma.size, ma->size);

      const std::size_t M = ma->get_num_base_elements(N, B);
      ASSERT_TRUE(M * ma->num_bits_per_base >= N * B);

      h_ma.clear();
      h_ma.resize(M);

      sycl_target->queue
          .memcpy(h_ma.data(), d_ma.d_masks, M * sizeof(MaskArrayBaseType))
          .wait_and_throw();
    }
  }
  sycl_target->free();
}
