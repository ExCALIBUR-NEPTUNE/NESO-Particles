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
