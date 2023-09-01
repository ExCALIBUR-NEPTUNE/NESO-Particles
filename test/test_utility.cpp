#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST(Utility, UniformWithinExtents) {

  {
    std::mt19937 rng = std::mt19937(std::random_device{}());
    const double extents[1] = {1.0};
    auto u0 = uniform_within_extents(1, 1, extents, rng);
    auto u1 = uniform_within_extents(1, 1, extents, rng);
    ASSERT_TRUE(u0[0][0] != u1[0][0]);
  }

  {
    const double extents[1] = {1.0};
    auto u0 = uniform_within_extents(1, 1, extents);
    auto u1 = uniform_within_extents(1, 1, extents);
    ASSERT_TRUE(u0[0][0] != u1[0][0]);
  }

  {
    std::mt19937 rng0 = std::mt19937(123);
    const double extents[1] = {1.0};
    auto u0 = uniform_within_extents(1, 1, extents, rng0);

    std::mt19937 rng1 = std::mt19937(123);
    auto u1 = uniform_within_extents(1, 1, extents, rng1);
    ASSERT_TRUE(u0[0][0] == u1[0][0]);
  }

  {
    std::mt19937 rng0 = std::mt19937(123);
    const double extents[3] = {1.0, 2.0, 3.0};
    auto u = uniform_within_extents(32, 3, extents, rng0);
    for (int ix = 0; ix < 32; ix++) {
      for (int dx = 0; dx < 3; dx++) {
        ASSERT_TRUE(u[dx][ix] >= 0.0);
        ASSERT_TRUE(u[dx][ix] <= extents[dx]);
      }
    }
  }
}

TEST(Utility, NormalDistribution) {

  {
    std::mt19937 rng = std::mt19937(std::random_device{}());
    auto u0 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0, rng);
    auto u1 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0, rng);
    ASSERT_TRUE(u0[0][0] != u1[0][0]);
  }

  {
    auto u0 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0);
    auto u1 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0);
    ASSERT_TRUE(u0[0][0] != u1[0][0]);
  }

  {
    std::mt19937 rng0 = std::mt19937(123);
    const double extents[1] = {1.0};
    auto u0 = NESO::Particles::normal_distribution(1, 1, 2.0, 3.0, rng0);

    std::mt19937 rng1 = std::mt19937(123);
    auto u1 = NESO::Particles::normal_distribution(1, 1, 2.0, 3.0, rng1);
    ASSERT_TRUE(u0[0][0] == u1[0][0]);
  }
}
