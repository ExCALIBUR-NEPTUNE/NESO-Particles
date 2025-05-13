#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, remove_particles) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto bb = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  ASSERT_EQ(bb->get_npart_local() + aa->get_npart_local(),
            A->get_npart_local());

  A->reset_version_tracker();
  A->remove_particles(aa);
  A->test_version_different();
  A->test_internal_state();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());
  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  ASSERT_EQ(bb->get_npart_local(), A->get_npart_local());
  ASSERT_EQ(aa->get_npart_local(), 0);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = id->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      ASSERT_TRUE(id->at(rowx, 0) % 2 == 1);
    }
    ASSERT_EQ(bb->get_npart_cell(cellx), A->get_npart_cell(cellx));
    ASSERT_EQ(aa->get_npart_cell(cellx), 0);
  }

  auto AA = std::make_shared<ParticleSubGroup>(A);

  A->reset_version_tracker();
  A->remove_particles(AA);
  A->test_version_different();
  A->test_init();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());
  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  ASSERT_EQ(A->get_npart_local(), 0);
  ASSERT_EQ(AA->get_npart_local(), 0);
  ASSERT_EQ(bb->get_npart_local(), 0);
  ASSERT_EQ(aa->get_npart_local(), 0);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(A->get_npart_cell(cellx), 0);
    ASSERT_EQ(AA->get_npart_cell(cellx), 0);
    ASSERT_EQ(bb->get_npart_cell(cellx), 0);
    ASSERT_EQ(aa->get_npart_cell(cellx), 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, clear) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto bb = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));
  auto ee = std::make_shared<ParticleSubGroup>(
      A, [=](auto /*ID*/) { return false; }, Access::read(Sym<INT>("ID")));

  auto AA = std::make_shared<ParticleSubGroup>(A);
  A->reset_version_tracker();
  A->clear();
  A->test_version_different();
  A->test_init();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());
  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  ASSERT_EQ(A->get_npart_local(), 0);
  ASSERT_EQ(AA->get_npart_local(), 0);
  ASSERT_EQ(bb->get_npart_local(), 0);
  ASSERT_EQ(aa->get_npart_local(), 0);
  ASSERT_EQ(ee->get_npart_local(), 0);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(A->get_npart_cell(cellx), 0);
    ASSERT_EQ(AA->get_npart_cell(cellx), 0);
    ASSERT_EQ(bb->get_npart_cell(cellx), 0);
    ASSERT_EQ(aa->get_npart_cell(cellx), 0);
    ASSERT_EQ(ee->get_npart_cell(cellx), 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, add_product_matrix) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto bb = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));
  auto ee = std::make_shared<ParticleSubGroup>(
      A, [=](auto /*ID*/) { return false; }, Access::read(Sym<INT>("ID")));

  const int npart_b_A = A->get_npart_local();
  const int npart_b_aa = aa->get_npart_local();
  const int npart_b_bb = bb->get_npart_local();

  auto product_spec = product_matrix_spec(ParticleProp(Sym<INT>("MARKER"), 1));
  auto pm = product_matrix(sycl_target, product_spec);
  pm->reset(1);

  A->reset_version_tracker();
  A->add_particles_local(pm);
  A->test_version_different();
  A->test_internal_state();

  ASSERT_EQ(A->get_npart_local(), npart_b_A + 1);
  ASSERT_EQ(bb->get_npart_local(), npart_b_bb);
  ASSERT_EQ(aa->get_npart_local(), npart_b_aa + 1);
  ASSERT_EQ(ee->get_npart_local(), 0);

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, add_particles_local_particle_group) {
  auto A = subgroup_test_common(10);
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  auto B =
      make_test_obj<ParticleGroup>(domain, A->get_particle_spec(), sycl_target);

  auto product_spec = product_matrix_spec(ParticleProp(Sym<INT>("MARKER"), 1));
  auto pm = product_matrix(sycl_target, product_spec);
  pm->reset(1);

  B->reset_version_tracker();
  B->add_particles_local(pm);
  B->test_version_different();
  B->test_internal_state();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->reset_version_tracker();
  A->add_particles_local(B);
  A->test_version_different();
  A->test_internal_state();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  B->free();
  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, is_particle_sub_group) {
  ParticleSubGroupSharedPtr sub_group;
  ParticleGroupSharedPtr group;
  ASSERT_TRUE(is_particle_sub_group(sub_group));
  ASSERT_FALSE(is_particle_sub_group(group));
}
