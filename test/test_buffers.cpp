#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

TEST(Buffer, host) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  for (std::size_t alignment : {0, 32}) {
    const int N = 15;
    std::vector<int> correct(N);
    for (int ix = 0; ix < N; ix++) {
      correct.at(ix) = ix;
    }

    BufferHost to_test{sycl_target, correct, alignment};

    EXPECT_EQ(to_test.size, N);
    EXPECT_EQ(to_test.size_bytes(), N * sizeof(int));

    for (int ix = 0; ix < N; ix++) {
      EXPECT_EQ(correct[ix], to_test.ptr[ix]);
    }

    std::vector<double> empty(0);
    BufferHost to_test_empty{sycl_target, empty, alignment};

    BufferHost to_test_vector{sycl_target, correct, alignment};
    for (int ix = 0; ix < N; ix++) {
      EXPECT_EQ(correct[ix], to_test_vector.ptr[ix]);
    }
  }
  sycl_target->free();
}

TEST(Buffer, device) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  for (std::size_t alignment : {0, 32}) {
    const int N = 15;
    std::vector<int> correct(N);
    std::vector<int> to_test(N);
    for (int ix = 0; ix < N; ix++) {
      correct.at(ix) = ix;
    }

    const std::size_t num_bytes = to_test.size() * sizeof(int);
    auto d_ptr =
        static_cast<int *>(sycl_target->malloc_device(num_bytes, alignment));
    BufferDevice buffer{sycl_target, correct, alignment};

    EXPECT_EQ(buffer.size, N);
    EXPECT_EQ(buffer.size_bytes(), N * sizeof(int));

    const auto k_to_test = buffer.ptr;
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
            d_ptr[idx] = k_to_test[idx];
          });
        })
        .wait_and_throw();

    sycl_target->queue.memcpy(to_test.data(), d_ptr, num_bytes)
        .wait_and_throw();
    for (int ix = 0; ix < N; ix++) {
      EXPECT_EQ(correct[ix], to_test[ix]);
    }

    std::vector<double> empty(0);
    BufferDevice to_test_empty{sycl_target, empty, alignment};

    for (int ix = 0; ix < N; ix++) {
      correct[ix] *= 2;
    }

    BufferDevice to_test_vector{sycl_target, correct, alignment};
    auto k_to_test_vector = to_test_vector.ptr;

    std::vector<int> to_test2(N);
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
            d_ptr[idx] = k_to_test_vector[idx];
          });
        })
        .wait_and_throw();

    sycl_target->queue.memcpy(to_test2.data(), d_ptr, num_bytes)
        .wait_and_throw();
    for (int ix = 0; ix < N; ix++) {
      EXPECT_EQ(correct[ix], to_test2[ix]);
    }

    auto to_test_get = to_test_vector.get();
    for (int ix = 0; ix < N; ix++) {
      ASSERT_NEAR(to_test_get.at(ix), correct.at(ix), 1.0e-15);
      to_test_get.at(ix) *= 4;
      correct.at(ix) *= 4;
    }

    to_test_vector.set(correct);
    to_test_get = to_test_vector.get();
    for (int ix = 0; ix < N; ix++) {
      ASSERT_NEAR(to_test_get.at(ix), correct.at(ix), 1.0e-15);
    }

    sycl_target->free(d_ptr);
  }
  sycl_target->free();
}

TEST(Buffer, device_host) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  for (std::size_t alignment : {0, 32}) {
    const int N = 15;
    std::vector<int> correct(N);
    std::vector<int> to_test(N);
    for (int ix = 0; ix < N; ix++) {
      correct.at(ix) = ix;
    }
    const std::size_t num_bytes = to_test.size() * sizeof(int);

    std::vector<int> host_buffer(N);
    auto h_ptr = host_buffer.data();
    auto d_ptr =
        static_cast<int *>(sycl_target->malloc_device(num_bytes, alignment));

    BufferDeviceHost buffer{sycl_target, correct, alignment};
    EXPECT_EQ(buffer.size, N);
    EXPECT_EQ(buffer.size_bytes(), N * sizeof(int));

    auto k_to_test = buffer.d_buffer.ptr;
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
            d_ptr[idx] = k_to_test[idx];
          });
        })
        .wait_and_throw();

    {
      sycl_target->queue.memcpy(h_ptr, d_ptr, num_bytes).wait_and_throw();
      for (int ix = 0; ix < N; ix++) {
        EXPECT_EQ(correct[ix], h_ptr[ix]);
        EXPECT_EQ(correct[ix], buffer.h_buffer.ptr[ix]);
      }
    }

    for (int ix = 0; ix < N; ix++) {
      buffer.h_buffer.ptr[ix] *= 2;
    }
    buffer.host_to_device();

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
            d_ptr[idx] = k_to_test[idx];
            k_to_test[idx] *= 4;
          });
        })
        .wait_and_throw();

    buffer.device_to_host();
    {
      sycl_target->queue.memcpy(h_ptr, d_ptr, num_bytes).wait_and_throw();
      for (int ix = 0; ix < N; ix++) {
        EXPECT_EQ(correct[ix] * 2, h_ptr[ix]);
        EXPECT_EQ(correct[ix] * 8, buffer.h_buffer.ptr[ix]);
      }
    }

    std::vector<double> empty(0);
    BufferDeviceHost to_test_empty{sycl_target, empty, alignment};

    for (int ix = 0; ix < N; ix++) {
      correct.at(ix) *= 2;
    }

    {
      BufferDeviceHost to_test_vector{sycl_target, correct, alignment};
      auto k_to_test_vector = to_test_vector.d_buffer.ptr;
      sycl_target->queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for<>(sycl::range<1>(N), [=](sycl::id<1> idx) {
              k_to_test_vector[idx] *= 2;
            });
          })
          .wait_and_throw();

      for (int ix = 0; ix < N; ix++) {
        EXPECT_EQ(correct[ix], to_test_vector.h_buffer.ptr[ix]);
      }
      to_test_vector.device_to_host();
      for (int ix = 0; ix < N; ix++) {
        EXPECT_EQ(correct[ix] * 2, to_test_vector.h_buffer.ptr[ix]);
      }
    }

    sycl_target->free(d_ptr);
  }
  sycl_target->free();
}

TEST(Buffer, realloc_host) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  for (std::size_t alignment : {0, 32}) {
    const int N = 15;
    std::vector<int> correct(N);
    for (int ix = 0; ix < N; ix++) {
      correct.at(ix) = ix;
    }

    BufferHost<int> b(sycl_target, correct, alignment);
    for (int ix = 0; ix < N; ix++) {
      ASSERT_EQ(correct[ix], b.ptr[ix]);
    }
    b.realloc(2 * N);
    ASSERT_EQ(b.size, 2 * N);

    for (int ix = 0; ix < N; ix++) {
      ASSERT_EQ(correct[ix], b.ptr[ix]);
    }

    for (int ix = 0; ix < N; ix++) {
      b.ptr[ix + N] = 2 * ix;
    }
    for (int ix = 0; ix < N; ix++) {
      ASSERT_EQ(correct[ix] * 2, b.ptr[ix + N]);
    }
  }
  sycl_target->free();
}

TEST(Buffer, realloc_device) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  for (std::size_t alignment : {0, 32}) {
    const int N = 15;
    std::vector<int> correct(N);
    std::vector<int> to_test(2 * N);
    for (int ix = 0; ix < N; ix++) {
      correct.at(ix) = ix;
    }

    BufferDevice<int> b(sycl_target, correct, alignment);
    sycl_target->queue.memcpy(to_test.data(), b.ptr, N * sizeof(int))
        .wait_and_throw();

    for (int ix = 0; ix < N; ix++) {
      ASSERT_EQ(correct[ix], to_test[ix]);
    }
    b.realloc(2 * N);
    ASSERT_EQ(b.size, 2 * N);

    sycl_target->queue.memcpy(to_test.data(), b.ptr, N * sizeof(int))
        .wait_and_throw();
    for (int ix = 0; ix < N; ix++) {
      ASSERT_EQ(correct[ix], to_test[ix]);
    }

    auto k_ptr = b.ptr;
    sycl_target->queue
        .parallel_for<>(
            sycl_target->device_limits.validate_range_global(sycl::range<1>(N)),
            [=](sycl::id<1> idx) { k_ptr[idx + N] = 2 * idx; })
        .wait_and_throw();

    sycl_target->queue.memcpy(to_test.data(), b.ptr, 2 * N * sizeof(int))
        .wait_and_throw();
    for (int ix = 0; ix < N; ix++) {
      ASSERT_EQ(correct[ix] * 2, to_test[ix + N]);
    }
  }
  sycl_target->free();
}
