
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

  particle_loop(
      aa,
      [=](auto INDEX, auto DATS) {
        for (int dx = 0; dx < ndim; dx++) {
          DATS.at_ephemeral(0, INDEX, dx) = DATS.at(1, dx);
        }
      },
      Access::read(ParticleLoopIndex{}),
      Access::write(sym_vector(aa, {Sym<REAL>("NORMAL"), Sym<REAL>("V")})))
      ->execute();

  particle_loop(
      aa,
      [=](auto INDEX, auto DATS) {
        for (int dx = 0; dx < ndim; dx++) {
          NESO_KERNEL_ASSERT(DATS.at_ephemeral(0, INDEX, dx) == DATS.at(1, dx),
                             k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}),
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

  ASSERT_FALSE(aa->invalidate_ephemeral_dats_if_required());

  particle_loop(
      A,
      [=](auto) {

      },
      Access::write(Sym<INT>("ID")))
      ->execute();

  ASSERT_TRUE(aa->invalidate_ephemeral_dats_if_required());
  ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
  ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));

  auto lambda_test_a = [&]() {
    aa->add_ephemeral_dat(Sym<REAL>("NORMAL"), 2);
    aa->add_ephemeral_dat(Sym<REAL>("INTERSECTION_POINT"), 4);
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));
    aa->add_ephemeral_dat(Sym<INT>("IDE"), 2);
    aa->add_ephemeral_dat(Sym<INT>("COMPOSITE_ID"), 1);
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("INTERSECTION_POINT")));
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<INT>("COMPOSITE_ID")));
  };

  auto lambda_test_b = [&]() {
    ASSERT_TRUE(aa->invalidate_ephemeral_dats_if_required());
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<REAL>("NORMAL")));
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<REAL>("INTERSECTION_POINT")));
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("IDE")));
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("COMPOSITE_ID")));
  };

  lambda_test_a();
  A->hybrid_move();
  lambda_test_b();

  lambda_test_a();
  A->cell_move();
  lambda_test_b();

  lambda_test_a();
  A->invalidate_group_version();
  lambda_test_b();

  aa->static_status(true);
  lambda_test_a();

  auto aa_empty = particle_sub_group(A, [=]() { return false; });
  aa_empty->add_ephemeral_dat(Sym<REAL>("FOO"), 2);

  sycl_target->free();
}

TEST(EphemeralDats, whole_group) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto aa = particle_sub_group(A);

  auto lambda_test_a = [&]() {
    aa->add_ephemeral_dat(Sym<REAL>("VCOPY"), 3);
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("VCOPY")));
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("IDCOPY")));

    aa->add_ephemeral_dat(Sym<INT>("IDCOPY"), 2);
    aa->add_ephemeral_dat(Sym<INT>("IDE"), 2);
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("VCOPY")));
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<INT>("IDCOPY")));
  };

  auto lambda_test_b = [&]() {
    ASSERT_TRUE(aa->invalidate_ephemeral_dats_if_required());
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<REAL>("VCOPY")));
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<INT>("IDCOPY")));
  };

  lambda_test_a();
  A->invalidate_group_version();
  lambda_test_b();

  lambda_test_a();

  particle_loop(
      aa,
      [=](auto V, auto VCOPY, auto ID) {
        VCOPY.at_ephemeral(0) = V.at(0);
        VCOPY.at_ephemeral(1) = V.at(1);
        ID.at_ephemeral(1, 0) = ID.at(0, 0);
      },
      Access::read(Sym<REAL>("V")), Access::write(Sym<REAL>("VCOPY")),
      Access::write(std::make_shared<SymVector<INT>>(
          A, std::vector({Sym<INT>("ID"), Sym<INT>("IDCOPY")}))))
      ->execute();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      aa,
      [=](auto V, auto VCOPY, auto ID) {
        NESO_KERNEL_ASSERT(VCOPY.at_ephemeral(0) == V.at(0), k_ep);
        NESO_KERNEL_ASSERT(VCOPY.at_ephemeral(1) == V.at(1), k_ep);
        NESO_KERNEL_ASSERT(ID.at_ephemeral(1, 0) == ID.at(0, 0), k_ep);
      },
      Access::read(Sym<REAL>("V")), Access::read(Sym<REAL>("VCOPY")),
      Access::read(std::make_shared<SymVector<INT>>(
          A, std::vector({Sym<INT>("ID"), Sym<INT>("IDCOPY")}))))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
}

TEST(EphemeralDats, static_invalidation) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto lambda_create = [&](auto &aa) {
    aa->add_ephemeral_dat(Sym<REAL>("VCOPY"), 3);
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("VCOPY")));
  };

  auto lambda_test_exists = [&](auto &aa) {
    ASSERT_TRUE(aa->contains_ephemeral_dat(Sym<REAL>("VCOPY")));
  };

  auto lambda_test_not_exists = [&](auto &aa) {
    ASSERT_FALSE(aa->contains_ephemeral_dat(Sym<REAL>("VCOPY")));
  };

  {
    auto aa = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));
    lambda_create(aa);
    lambda_test_exists(aa);
    particle_loop(
        aa, [=](auto ID) { ID.at(0) += 2; }, Access::write(Sym<INT>("ID")))
        ->execute();
    lambda_test_not_exists(aa);

    lambda_create(aa);
    lambda_test_exists(aa);
    A->invalidate_group_version();
    lambda_test_not_exists(aa);
  }

  {
    auto aa = static_particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));
    lambda_create(aa);
    lambda_test_exists(aa);
    particle_loop(
        aa, [=](auto ID) { ID.at(0) += 2; }, Access::write(Sym<INT>("ID")))
        ->execute();
    lambda_test_exists(aa);

    lambda_create(aa);
    lambda_test_exists(aa);
    A->invalidate_group_version();
    lambda_test_not_exists(aa);
  }

  sycl_target->free();
  A->domain->mesh->free();
}
