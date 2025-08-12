#include "include/test_neso_particles.hpp"

TEST(Algorithms, unseen_value_extractor) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  A->add_particle_dat(Sym<INT>("FOO"), 2);
  particle_loop(
      A,
      [=](auto ID, auto FOO) {
        FOO.at(0) = ID.at(0);
        FOO.at(1) = ID.at(0);
      },
      Access::read(Sym<INT>("ID")), Access::write(Sym<INT>("FOO")))
      ->execute();

  std::set<INT> seen_values;

  UnseenValueExtractor uve(sycl_target);

  auto u0 = uve.extract(A, Sym<INT>("FOO"), 0, false);
  for (auto ux : u0) {
    ASSERT_EQ(seen_values.count(ux), 0);
    seen_values.insert(ux);
  }

  auto u1 = uve.extract(A, Sym<INT>("FOO"), 1, false);
  ASSERT_EQ(u1.size(), 0);

  auto aa = particle_sub_group(A);

  aa->add_ephemeral_dat(Sym<INT>("BAR"), 1);

  particle_loop(
      aa, [=](auto ID, auto BAR) { BAR.at_ephemeral(0) = ID.at(0); },
      Access::read(Sym<INT>("ID")), Access::write(Sym<INT>("BAR")))
      ->execute();

  auto u2 = uve.extract(aa, Sym<INT>("BAR"), 0, true);
  ASSERT_EQ(u2.size(), 0);

  sycl_target->free();
  A->domain->mesh->free();
}
