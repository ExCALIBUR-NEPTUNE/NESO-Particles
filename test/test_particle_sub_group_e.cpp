#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, add_particles_local_particle_sub_group) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto odd = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto even = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  auto B =
      make_test_obj<ParticleGroup>(domain, A->get_particle_spec(), sycl_target);

  auto bb = std::make_shared<ParticleSubGroup>(
      B, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  B->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<INT>("FOO"), 3), cell_count));

  const int npart_local = A->get_npart_local();
  const int npart_local_even = even->get_npart_local();
  const int npart_local_odd = odd->get_npart_local();
  ASSERT_EQ(npart_local, npart_local_even + npart_local_odd);

  std::vector<int> npart_cell(cell_count);
  std::vector<int> npart_cell_even(cell_count);
  std::vector<int> npart_cell_odd(cell_count);

  std::map<int, std::set<INT>> cell_to_ids;
  std::set<INT> ids;

  for (int cx = 0; cx < cell_count; cx++) {
    npart_cell.at(cx) = A->get_npart_cell(cx);
    npart_cell_even.at(cx) = even->get_npart_cell(cx);
    npart_cell_odd.at(cx) = odd->get_npart_cell(cx);

    const int npart_A = A->get_npart_cell(cx);
    auto A_ID = A->get_cell(Sym<INT>("ID"), cx);
    for (int rowx = 0; rowx < npart_A; rowx++) {
      const INT id = A_ID->at(rowx, 0);
      cell_to_ids[cx].insert(id);
      ids.insert(id);
    }
    ASSERT_EQ(cell_to_ids[cx].size(), npart_cell.at(cx));
  }
  ASSERT_EQ(ids.size(), npart_local);

  B->reset_version_tracker();
  B->add_particles_local(even);
  B->test_version_different();
  B->test_internal_state();

  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  for (int cx = 0; cx < cell_count; cx++) {
    const int npart = B->get_npart_cell(cx);
    ASSERT_EQ(npart, npart_cell_even.at(cx));
    auto B_ID = B->get_cell(Sym<INT>("ID"), cx);
    for (int rowx = 0; rowx < npart; rowx++) {
      ASSERT_TRUE(B_ID->at(rowx, 0) % 2 == 0);
    }
  }

  A->reset_version_tracker();
  A->remove_particles(even);
  A->test_version_different();
  A->test_internal_state();

  EXPECT_TRUE(even->create_if_required());
  EXPECT_FALSE(even->create_if_required());
  ASSERT_EQ(even->get_npart_local(), 0);
  ASSERT_EQ(npart_local_odd, A->get_npart_local());
  ASSERT_EQ(npart_local_even, B->get_npart_local());

  std::set<INT> ids_to_test;
  for (int cx = 0; cx < cell_count; cx++) {
    const int npart_A = A->get_npart_cell(cx);
    const int npart_B = B->get_npart_cell(cx);

    ASSERT_EQ(npart_A, npart_cell_odd.at(cx));
    ASSERT_EQ(npart_B, npart_cell_even.at(cx));

    auto B_ID = B->get_cell(Sym<INT>("ID"), cx);
    auto A_ID = A->get_cell(Sym<INT>("ID"), cx);
    std::set<INT> cell_ids;

    for (int rowx = 0; rowx < npart_A; rowx++) {
      const INT id = A_ID->at(rowx, 0);
      ASSERT_TRUE(-1 < id);
      ASSERT_TRUE(id % 2 == 1);
      cell_ids.insert(id);
      ids_to_test.insert(id);
    }
    for (int rowx = 0; rowx < npart_B; rowx++) {
      const INT id = B_ID->at(rowx, 0);
      ASSERT_TRUE(-1 < id);
      ASSERT_TRUE(id % 2 == 0);
      cell_ids.insert(id);
      ids_to_test.insert(id);
    }
    ASSERT_TRUE(cell_ids == cell_to_ids[cx]);
  }
  ASSERT_EQ(ids_to_test.size(), ids.size());
  ASSERT_TRUE(ids_to_test == ids);

  for (int cx = 0; cx < cell_count; cx++) {
    const int npart = B->get_npart_cell(cx);
    auto FOO = B->get_cell(Sym<INT>("FOO"), cx);
    for (int rx = 0; rx < npart; rx++) {
      for (int dx = 0; dx < 3; dx++) {
        ASSERT_EQ(FOO->at(rx, dx), 0);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, sub_sub_group) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto mod2 = std::make_shared<TestParticleSubGroup>(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  auto mod4 = std::make_shared<TestParticleSubGroup>(
      std::dynamic_pointer_cast<ParticleSubGroup>(mod2),
      [=](auto ID) { return ((ID[0] % 4) == 0); },
      Access::read(Sym<INT>("ID")));

  ASSERT_EQ(mod2->get_particle_group(), mod4->get_particle_group());

  std::vector<INT> cells;
  std::vector<INT> layers;
  auto num_particles = mod2->test_get_cells_layers(cells, layers);
  std::set<std::pair<INT, INT>> correct, to_test;
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);
    auto id = A->get_cell(Sym<INT>("ID"), cellx)->at(layerx, 0);
    ASSERT_TRUE(id % 2 == 0);
    if (id % 4 == 0) {
      correct.insert({cellx, layerx});
    }
  }

  num_particles = mod4->test_get_cells_layers(cells, layers);
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);
    auto id = A->get_cell(Sym<INT>("ID"), cellx)->at(layerx, 0);
    ASSERT_TRUE(id % 4 == 0);
    to_test.insert({cellx, layerx});
  }
  ASSERT_EQ(to_test, correct);

  to_test.clear();
  auto AA = particle_sub_group(A);
  auto mod42 = std::make_shared<TestParticleSubGroup>(
      AA, [=](auto ID) { return ((ID[0] % 4) == 0); },
      Access::read(Sym<INT>("ID")));

  num_particles = mod42->test_get_cells_layers(cells, layers);
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);
    auto id = A->get_cell(Sym<INT>("ID"), cellx)->at(layerx, 0);
    ASSERT_TRUE(id % 4 == 0);
    to_test.insert({cellx, layerx});
  }
  ASSERT_EQ(to_test, correct);

  auto m0 = particle_sub_group(
      A, [=](auto /*M*/) { return true; }, Access::read(Sym<INT>("MARKER")));

  auto e0 = particle_sub_group(
      m0, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  EXPECT_TRUE(e0->create_if_required());
  EXPECT_FALSE(e0->create_if_required());

  particle_loop(
      A, [=](auto M) { M.at(0) += 1; }, Access::write(Sym<INT>("MARKER")))
      ->execute();

  EXPECT_TRUE(e0->create_if_required());
  EXPECT_FALSE(e0->create_if_required());

  A->free();
  sycl_target->free();
  mesh->free();
}
