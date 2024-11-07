#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>

using namespace NESO::Particles;

TEST(SYCLTarget, print_device_info) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  sycl_target->print_device_info();
  sycl_target->free();
}

TEST(SYCLTarget, joint_exclusive_scan) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int N = 21241;
  const std::size_t group_size =
      std::min(static_cast<std::size_t>(
                   sycl_target->device
                       .get_info<sycl::info::device::max_work_group_size>()),
               static_cast<std::size_t>(N));
  ASSERT_TRUE(group_size >= 1);

  std::vector<int> h_src(N);
  std::vector<int> h_correct(N);
  std::iota(h_src.begin(), h_src.end(), 0);
  std::exclusive_scan(h_src.begin(), h_src.end(), h_correct.begin(), 0);

  BufferDevice d_src(sycl_target, h_src);
  BufferDevice d_dst(sycl_target, h_src);

  joint_exclusive_scan(sycl_target, N, d_src.ptr, d_dst.ptr).wait_and_throw();

  auto h_to_test = d_dst.get();

  EXPECT_EQ(h_to_test, h_correct);

  sycl_target->free();
}

TEST(SYCLTarget, parameters) {
  const std::size_t local_size =
      get_env_size_t("NESO_PARTICLES_LOOP_LOCAL_SIZE", 32);
  const std::size_t nbin = get_env_size_t("NESO_PARTICLES_LOOP_NBIN", 4);

  auto p_local_size = std::make_shared<SizeTParameter>(local_size);
  auto p_nbin = std::make_shared<SizeTParameter>(nbin);

  Parameters p;
  p.set("LOOP_LOCAL_SIZE", p_local_size);
  p.set("LOOP_NBIN", p_nbin);

  EXPECT_EQ(p.get<SizeTParameter>("LOOP_LOCAL_SIZE")->value, local_size);
  EXPECT_EQ(p.get<SizeTParameter>("LOOP_NBIN")->value, nbin);
}

TEST(SYCLTarget, matrix_transpose) {

  auto lambda_transpose = [](const std::size_t num_rows,
                             const std::size_t num_cols, auto &h_src,
                             auto &h_dst) {
    for (std::size_t rowx = 0; rowx < num_rows; rowx++) {
      for (std::size_t colx = 0; colx < num_cols; colx++) {
        h_dst.at(colx * num_rows + rowx) = h_src.at(rowx * num_cols + colx);
      }
    }
  };

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  {
    std::vector<int> h_src_simple = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> h_dst_correct = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<int> h_dst_to_test(8);
    lambda_transpose(2, 4, h_src_simple, h_dst_to_test);
    ASSERT_EQ(h_dst_correct, h_dst_to_test);

    BufferDevice<int> d_src(sycl_target, h_src_simple);
    BufferDevice<int> d_dst(sycl_target, h_src_simple);

    matrix_transpose(sycl_target, 2, 4, d_src.ptr, d_dst.ptr).wait_and_throw();
    std::fill(h_dst_to_test.begin(), h_dst_to_test.end(), 0);
    d_dst.get(h_dst_to_test);
    ASSERT_EQ(h_dst_correct, h_dst_to_test);
  }

  {
    const std::size_t num_rows = 7919;
    const std::size_t num_cols = 1483;
    std::vector<REAL> h_src(num_rows * num_cols);
    std::vector<REAL> h_correct(num_rows * num_cols);
    std::iota(h_src.begin(), h_src.end(), 1.0);
    lambda_transpose(num_rows, num_cols, h_src, h_correct);
    BufferDevice<REAL> d_src(sycl_target, h_src);
    BufferDevice<REAL> d_dst(sycl_target, h_src);
    matrix_transpose(sycl_target, num_rows, num_cols, d_src.ptr, d_dst.ptr)
        .wait_and_throw();
    auto h_to_test = d_dst.get();
    ASSERT_EQ(h_correct, h_to_test);
  }
}
