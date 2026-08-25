#include "include/test_neso_particles.hpp"

TEST(Algorithms, unseen_value_extractor_unknown_bounds) {
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
  ASSERT_FALSE(uve.using_known_bounds());

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

TEST(Algorithms, unseen_value_extractor_known_bounds) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto la_min = std::make_shared<NDLocalArray<INT, 1>>(sycl_target, 1);
  auto la_max = std::make_shared<NDLocalArray<INT, 1>>(sycl_target, 1);

  la_min->fill(std::numeric_limits<INT>::max());
  la_max->fill(std::numeric_limits<INT>::lowest());

  particle_loop(
      A,
      [=](auto ID, auto LA_MAX, auto LA_MIN) {
        LA_MAX.fetch_max(0, ID.at(0));
        LA_MIN.fetch_min(0, ID.at(0));
      },
      Access::read(Sym<INT>("ID")), Access::max(la_max), Access::min(la_min))
      ->execute();

  const INT bound_lower = la_min->get().at(0);
  const INT bound_upper = la_max->get().at(0);

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

  UnseenValueExtractor uve(sycl_target, bound_lower, bound_upper);
  ASSERT_TRUE(uve.using_known_bounds());

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
