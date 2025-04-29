#include "include/test_neso_particles.hpp"

TEST(EphemeralDats, create) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  const int cell_count = cell_count_t;

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  sycl_target->free();
}
