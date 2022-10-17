#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

TEST(ErrorPropagate, Flag) {
  SYCLTarget sycl_target{0, MPI_COMM_WORLD};

  // create an object to track that an error should be thrown
  ErrorPropagate ep(sycl_target);

  auto k_ep = ep.device_ptr();
  // get the kernel parameter

  sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(8), [=](sycl::id<1> idx) {
          // throw an error
          NESO_KERNEL_ASSERT(false, k_ep);
        });
      })
      .wait_and_throw();

  ASSERT_EQ(ep.get_flag(), 8);
  // ep.check_and_throw("This should abort.");
}
