
#include "include/test_neso_particles.hpp"

TEST(EphemeralDats, create) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  constexpr int ndim = 2;

  const int cell_count = cell_count_t;

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));

  aa->add_ephemeral_dat(Sym<REAL>("NORMAL"), 2);
  ASSERT_FALSE(aa->create_if_required());
  ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
  ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("ID")));

  aa->add_ephemeral_dat(Sym<INT>("ID"), 2);
  ASSERT_FALSE(aa->create_if_required());
  ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
  ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<INT>("ID")));

  particle_loop(
      aa,
      [=](auto NORMAL, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          NORMAL.at_ephemeral(dx) = V.at(dx);
        }
      },
      Access::write(Sym<REAL>("NORMAL")), Access::read(Sym<REAL>("V")))
      ->execute();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      aa,
      [=](auto DATS) {
        for (int dx = 0; dx < ndim; dx++) {
          NESO_KERNEL_ASSERT(DATS.at_ephemeral(0, dx) == DATS.at(1, dx), k_ep);
        }
      },
      Access::read(sym_vector(aa, {Sym<REAL>("NORMAL"), Sym<REAL>("V")})))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
}
