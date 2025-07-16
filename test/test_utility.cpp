#include <gtest/gtest.h>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST(Utility, uniform_within_extents) {

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
    EXPECT_TRUE(u0[0][0] != u1[0][0]);
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

TEST(Utility, normal_distribution) {

  {
    std::mt19937 rng = std::mt19937(std::random_device{}());
    auto u0 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0, rng);
    auto u1 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0, rng);
    ASSERT_TRUE(u0[0][0] != u1[0][0]);
  }

  {
    auto u0 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0);
    auto u1 = NESO::Particles::normal_distribution(1, 1, 1.0, 1.0);
    EXPECT_TRUE(u0[0][0] != u1[0][0]);
  }

  {
    std::mt19937 rng0 = std::mt19937(123);
    auto u0 = NESO::Particles::normal_distribution(1, 1, 2.0, 3.0, rng0);

    std::mt19937 rng1 = std::mt19937(123);
    auto u1 = NESO::Particles::normal_distribution(1, 1, 2.0, 3.0, rng1);
    ASSERT_TRUE(u0[0][0] == u1[0][0]);
  }
}

TEST(Utility, decomp_1d) {

  std::size_t N_compute_units = 7;
  std::size_t N_work_items = 31;

  for (std::size_t ix = 0; ix < N_work_items; ix++) {
    const std::size_t computed_work_unit =
        get_decomp_1d_inverse(N_compute_units, N_work_items, ix);

    std::size_t rstart, rend;
    get_decomp_1d(N_compute_units, N_work_items, computed_work_unit, &rstart,
                  &rend);

    ASSERT_TRUE(rstart <= ix);
    ASSERT_TRUE(ix < rend);
  }

  N_compute_units = 8;
  N_work_items = 32;

  for (std::size_t ix = 0; ix < N_work_items; ix++) {
    const std::size_t computed_work_unit =
        get_decomp_1d_inverse(N_compute_units, N_work_items, ix);

    std::size_t rstart, rend;
    get_decomp_1d(N_compute_units, N_work_items, computed_work_unit, &rstart,
                  &rend);

    ASSERT_TRUE(rstart <= ix);
    ASSERT_TRUE(ix < rend);
  }
}

TEST(Utility, flatten_map) {

  std::set<int> s, t;
  std::map<int, int> m;
  for (int ix = 0; ix < 37; ix++) {
    s.insert(ix + 100);
    m[ix] = ix + 100;
  }

  auto n = flatten_map(m);
  for (int nx : n) {
    t.insert(nx);
  }

  ASSERT_EQ(s, t);
}
