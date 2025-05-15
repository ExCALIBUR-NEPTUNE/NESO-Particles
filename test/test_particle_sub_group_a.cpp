#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, version_tracker) {

  ParticleDatVersionT v0;
  ParticleDatVersionT v1;
  EXPECT_FALSE(v0 < v1);
  ParticleDatVersionT r0(Sym<REAL>("A"));
  ParticleDatVersionT r1(Sym<REAL>("A"));
  ParticleDatVersionT r2(Sym<REAL>("B"));
  ParticleDatVersionT i0(Sym<INT>("A"));
  ParticleDatVersionT i1(Sym<INT>("A"));
  ParticleDatVersionT i2(Sym<INT>("B"));
  EXPECT_TRUE(v0 < r0);
  EXPECT_TRUE(v0 < i0);
  EXPECT_FALSE(r0 < i0);
  EXPECT_FALSE(r0 < r1);
  EXPECT_TRUE(r0 < r2);
  EXPECT_FALSE(i0 < i1);
  EXPECT_TRUE(i0 < i2);
  EXPECT_FALSE(r0 < i0);
  EXPECT_TRUE(i0 < r0);

  ParticleDatVersionT v2;
  v2 = Sym<INT>("C");
  ASSERT_TRUE(v2.index == 0);
  ASSERT_TRUE(v2.si.name == "C");
  ParticleDatVersionT v3;
  v3 = Sym<REAL>("C");
  ASSERT_TRUE(v3.index == 1);
  ASSERT_TRUE(v3.sr.name == "C");
  v2 = v3;
  ASSERT_TRUE(v2.index == 1);
  ASSERT_TRUE(v2.sr.name == "C");
  v3 = Sym<INT>("D");
  ASSERT_TRUE(v3.index == 0);
  ASSERT_TRUE(v3.si.name == "D");
}

TEST(ParticleSubGroup, selector) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  TestParticleSubGroup aa(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  aa.create();

  std::vector<INT> cells;
  std::vector<INT> layers;

  const int num_particles = aa.test_get_cells_layers(cells, layers);
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);

    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    EXPECT_TRUE((*id)[0][layerx] % 2 == 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, selector_get_even_id) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto s = std::make_shared<TestSubGroupSelector>(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  int cell_count = A->domain->mesh->get_cell_count();
  for (int cx = 0; cx < cell_count; cx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cx);
    for (int rx = 0; rx < ID->nrow; rx++) {
      if (ID->at(rx, 0) % 2 == 0) {
        cells.push_back(cx);
        layers.push_back(rx);
      }
    }
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, selector_get_even_id_even_cell) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto aa = particle_sub_group(
      A, [=](auto CELL_ID) { return ((CELL_ID.at(0) % 2) == 0); },
      Access::read(Sym<INT>("CELL_ID")));

  auto s = std::make_shared<TestSubGroupSelector>(
      aa, [=](auto ID) { return ((ID.at(0) % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  int cell_count = A->domain->mesh->get_cell_count();
  for (int cx = 0; cx < cell_count; cx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cx);
    auto CELL_ID = A->get_cell(Sym<INT>("CELL_ID"), cx);
    for (int rx = 0; rx < ID->nrow; rx++) {
      if ((ID->at(rx, 0) % 2 == 0) && (CELL_ID->at(rx, 0) % 2 == 0)) {
        cells.push_back(cx);
        layers.push_back(rx);
      }
    }
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, selector_get_cell) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  int cell_count = A->domain->mesh->get_cell_count();
  int cx = cell_count - 1;

  auto s = std::make_shared<TestCellSubGroupSelector>(A, cx);

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  const int nrow = A->get_npart_cell(cx);
  for (int rx = 0; rx < nrow; rx++) {
    cells.push_back(cx);
    layers.push_back(rx);
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, selector_get_cell_even_id) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  int cell_count = A->domain->mesh->get_cell_count();
  int cx = cell_count - 1;

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ((ID.at(0) % 2) == 0); },
      Access::read(Sym<INT>("ID")));
  auto s = std::make_shared<TestCellSubGroupSelector>(aa, cx);

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  const int nrow = A->get_npart_cell(cx);
  auto ID = A->get_cell(Sym<INT>("ID"), cx);
  for (int rx = 0; rx < nrow; rx++) {
    if (ID->at(rx, 0) % 2 == 0) {
      cells.push_back(cx);
      layers.push_back(rx);
    }
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, particle_loop) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] == 2; }, Access::read(Sym<INT>("ID")));

  GlobalArray<int> counter(sycl_target, 1, 0);

  auto pl_counter = particle_loop(
      aa, [=](Access::GlobalArray::Add<int> G1) { G1.add(0, 1); },
      Access::add(counter));

  pl_counter->execute();

  auto vector_c = counter.get();
  EXPECT_EQ(vector_c.at(0), 1);

  auto bb = std::make_shared<TestParticleSubGroup>(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  LocalArray<int> counter2(sycl_target, 2, 0);

  auto pl_counter2 = particle_loop(
      std::dynamic_pointer_cast<ParticleSubGroup>(bb),
      [=](auto G1, auto ID, auto MARKER) {
        G1.fetch_add(0, 1);
        G1.fetch_add(1, ID[0]);
        MARKER[0] = 1;
      },
      Access::add(counter2), Access::read(Sym<INT>("ID")),
      Access::write(Sym<INT>("MARKER")));

  pl_counter2->execute();
  auto vector_b = counter2.get();

  std::vector<INT> cells;
  std::vector<INT> layers;
  const int num_particles = bb->test_get_cells_layers(cells, layers);

  EXPECT_EQ(num_particles, vector_b.at(0));

  int id_counter = 0;
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);

    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    id_counter += (*id)[0][layerx];
  }

  EXPECT_EQ(id_counter, vector_b.at(1));

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    const int nrow = id->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      if ((*id)[0][rowx] % 2 == 0) {
        EXPECT_EQ((*marker)[0][rowx], 1);
      } else {
        EXPECT_EQ((*marker)[0][rowx], 0);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
