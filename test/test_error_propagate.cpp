#include <gtest/gtest.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

// test that int atomics are functional
TEST(ErrorPropagate, atomics) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  const int N = 1024;
  BufferDeviceHost<int> dh_a(sycl_target, 2);
  dh_a.h_buffer.ptr[0] = 0;
  dh_a.h_buffer.ptr[1] = 0;
  dh_a.host_to_device();
  auto k_ptr = dh_a.d_buffer.ptr;

  sycl_target->queue
      .submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::item<1> id) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              remove_count_atomic{k_ptr[0]};
          remove_count_atomic.fetch_add(1);
          if (id == 0) {
            k_ptr[1] = 42;
          }
        });
      })
      .wait_and_throw();

  dh_a.device_to_host();

  // test kernel actually ran
  ASSERT_EQ(dh_a.h_buffer.ptr[1], 42);
  // test atomics work
  ASSERT_EQ(dh_a.h_buffer.ptr[0], 1024);

  mesh->free();
}

TEST(ErrorPropagate, flag) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  // create an object to track that an error should be thrown
  ErrorPropagate ep(sycl_target);

  auto k_ep = ep.device_ptr();
  // get the kernel parameter

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(8), [=](sycl::id<1> idx) {
          // throw an error
          NESO_KERNEL_ASSERT(false, k_ep);
        });
      })
      .wait_and_throw();

  ASSERT_EQ(ep.get_flag(), 8);
  // ep.check_and_throw("This should abort.");
  ep.reset();
  ASSERT_EQ(ep.get_flag(), 0);
}
