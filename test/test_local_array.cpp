#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

TEST(LocalArray, get_set) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 6151;
  std::vector<REAL> d0(N);
  std::iota(d0.begin(), d0.end(), 0);
  LocalArray<REAL> l0(sycl_target, d0);
  std::fill(d0.begin(), d0.end(), 0);
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(0, d0[ix]);
  }
  l0.get(d0);
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(ix, d0[ix]);
    d0[ix] *= 2;
  }
  l0.set(d0);
  std::fill(d0.begin(), d0.end(), 0);
  l0.get(d0);
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(ix * 2, d0[ix]);
  }
}
