
#include "include/test_neso_particles.hpp"

TEST(EphemeralDats, base) {
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
  ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));

  aa->add_ephemeral_dat(Sym<INT>("IDE"), 2);
  ASSERT_FALSE(aa->create_if_required());
  ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
  ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));

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

  ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));

  particle_loop(
      aa,
      [=](auto INDEX, auto ID, auto IDE) {
        IDE.at_ephemeral(0) = INDEX.layer;
        IDE.at_ephemeral(1) = ID.at(0);
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("ID")),
      Access::write(Sym<INT>("IDE")))
      ->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cellx);
    auto IDE = aa->get_ephemeral_dat(Sym<INT>("IDE"))->cell_dat.get_cell(cellx);
    const int nrow = IDE->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT layer = IDE->at(rowx, 0);
      const INT correct_id = ID->at(layer, 0);
      const INT to_test_id = IDE->at(rowx, 1);
      ASSERT_EQ(correct_id, to_test_id);
    }
  }

  particle_loop(
      A,
      [=](auto) {

      },
      Access::write(Sym<INT>("ID")))
      ->execute();

  ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
  ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));

  sycl_target->free();
}
