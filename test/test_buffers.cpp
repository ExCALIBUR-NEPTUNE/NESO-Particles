#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

TEST(Buffer, Host) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 15;
  std::vector<int> correct(N);
  for (int ix = 0; ix < N; ix++) {
    correct.at(ix) = ix;
  }

  BufferHost to_test{sycl_target, correct};

  EXPECT_EQ(to_test.size, N);
  EXPECT_EQ(to_test.size_bytes(), N * sizeof(int));

  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(correct[ix], to_test.ptr[ix]);
  }

  std::vector<double> empty(0);
  BufferHost to_test_empty{sycl_target, empty};
}

TEST(Buffer, Device) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 15;
  std::vector<int> correct(N);
  std::vector<int> to_test(N);
  for (int ix = 0; ix < N; ix++) {
    correct.at(ix) = ix;
  }
  sycl::buffer<int, 1> b_to_test(to_test.data(), to_test.size());

  BufferHost buffer{sycl_target, correct};

  EXPECT_EQ(buffer.size, N);
  EXPECT_EQ(buffer.size_bytes(), N * sizeof(int));

  const auto k_to_test = buffer.ptr;
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        auto a_to_test = b_to_test.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
          a_to_test[idx] = k_to_test[idx];
        });
      })
      .wait_and_throw();

  auto h_to_test = b_to_test.get_host_access();
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(correct[ix], h_to_test[ix]);
  }

  std::vector<double> empty(0);
  BufferDevice to_test_empty{sycl_target, empty};
}

TEST(Buffer, DeviceHost) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 15;
  std::vector<int> correct(N);
  std::vector<int> to_test(N);
  for (int ix = 0; ix < N; ix++) {
    correct.at(ix) = ix;
  }
  sycl::buffer<int, 1> b_to_test(to_test.data(), to_test.size());

  BufferDeviceHost buffer{sycl_target, correct};
  EXPECT_EQ(buffer.size, N);
  EXPECT_EQ(buffer.size_bytes(), N * sizeof(int));

  auto k_to_test = buffer.d_buffer.ptr;
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        auto a_to_test = b_to_test.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
          a_to_test[idx] = k_to_test[idx];
        });
      })
      .wait_and_throw();

  {
    auto h_to_test = b_to_test.get_host_access();
    for (int ix = 0; ix < N; ix++) {
      EXPECT_EQ(correct[ix], h_to_test[ix]);
      EXPECT_EQ(correct[ix], buffer.h_buffer.ptr[ix]);
    }
  }

  for (int ix = 0; ix < N; ix++) {
    buffer.h_buffer.ptr[ix] *= 2;
  }
  buffer.host_to_device();

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        auto a_to_test = b_to_test.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
          a_to_test[idx] = k_to_test[idx];
          k_to_test[idx] *= 4;
        });
      })
      .wait_and_throw();

  buffer.device_to_host();
  {
    auto h_to_test = b_to_test.get_host_access();
    for (int ix = 0; ix < N; ix++) {
      EXPECT_EQ(correct[ix] * 2, h_to_test[ix]);
      EXPECT_EQ(correct[ix] * 8, buffer.h_buffer.ptr[ix]);
    }
  }

  std::vector<double> empty(0);
  BufferDeviceHost to_test_empty{sycl_target, empty};
}
