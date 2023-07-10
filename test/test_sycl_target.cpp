#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>

using namespace NESO::Particles;

TEST(SYCLTarget, print_device_info) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  sycl_target->print_device_info();
}
