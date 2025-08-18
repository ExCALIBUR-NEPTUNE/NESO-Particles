#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, creating) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return (ID[0] % 2) == 0; },
      Access::read(Sym<INT>("ID")));

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->cell_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->local_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->global_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->hybrid_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  auto remover = std::make_shared<ParticleRemover>(A->sycl_target);
  const int npart0 = A->get_npart_local();
  remover->remove(A, A->get_dat(Sym<INT>("ID")), 1);
  const int npart1 = A->get_npart_local();

  if (npart0 != npart1) {
    EXPECT_TRUE(aa->create_if_required());
  }
  EXPECT_FALSE(aa->create_if_required());

  auto la = std::make_shared<LocalArray<INT>>(sycl_target, 1);
  particle_loop(
      aa, [=](auto LA) { LA.fetch_add(0, 1); }, Access::add(la))
      ->execute();
  auto lav = la->get();
  const int npart_local = lav.at(0);
  int npart_min;
  MPICHK(MPI_Allreduce(&npart_local, &npart_min, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD));
  if (npart_min == 0) {
    return;
  }

  auto p0 = particle_loop(
      A, [](auto /*ID*/, auto V) { V[0] += 0.0001; },
      Access::read(Sym<INT>("ID")), Access::write(Sym<REAL>("V")));
  p0->execute();
  EXPECT_FALSE(aa->create_if_required());

  auto p1 = particle_loop(
      aa, [](auto /*ID*/, auto V) { V[0] += 0.0001; },
      Access::read(Sym<INT>("ID")), Access::write(Sym<REAL>("V")));
  p1->execute();
  EXPECT_FALSE(aa->create_if_required());

  p1->execute();
  p1->execute();
  EXPECT_FALSE(aa->create_if_required());

  auto p2 = particle_loop(
      A, [](auto ID) { ID[0] += 2; }, Access::write(Sym<INT>("ID")));
  p2->execute();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  std::vector<Sym<INT>> sym_vector_id = {Sym<INT>("ID")};
  aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID.at(0, 0) % 2 == 0; },
      Access::read(sym_vector(A, sym_vector_id)));

  auto p3 = particle_loop(
      aa, [](auto ID) { ID[0] += 2; }, Access::write(Sym<INT>("ID")));
  p3->execute();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  ParticleSet distribution(1, A->particle_spec);
  A->add_particles_local(distribution);

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  // This sets the P pointer to be "possibly cached"
  A->get_dat(Sym<REAL>("P"))->cell_dat.device_ptr();
  EXPECT_FALSE(aa->create_if_required());

  // This sets the ID pointer to be "possibly cached"
  A->get_dat(Sym<INT>("ID"))->cell_dat.device_ptr();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_TRUE(aa->create_if_required());

  EXPECT_EQ(aa->static_status(), false);
  EXPECT_EQ(aa->static_status(false), false);
  EXPECT_EQ(aa->static_status(true), true);
  EXPECT_EQ(aa->static_status(), true);
  EXPECT_TRUE(aa->is_valid());

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, static_valid) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto lambda_make_aa = [&]() {
    return static_particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));
  };

  auto aa = lambda_make_aa();
  EXPECT_TRUE(aa->is_valid());

  particle_loop(
      aa, [=](auto ID) { ID.at(0) += 2; }, Access::write(Sym<INT>("ID")))
      ->execute();
  EXPECT_TRUE(aa->is_valid());

  aa = lambda_make_aa();
  A->cell_move();
  EXPECT_TRUE(!aa->is_valid());

  aa = lambda_make_aa();
  A->hybrid_move();
  EXPECT_TRUE(!aa->is_valid());

  aa = lambda_make_aa();
  A->add_particles_local(aa);
  EXPECT_TRUE(!aa->is_valid());

  auto bb = static_particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 4 == 0; },
      Access::read(Sym<INT>("ID")));
  aa = lambda_make_aa();
  A->remove_particles(bb);
  EXPECT_TRUE(!aa->is_valid());

  A->free();
  sycl_target->free();
  mesh->free();
}
