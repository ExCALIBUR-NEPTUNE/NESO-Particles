#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, particle_loop_index) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  auto pl_reset = particle_loop(
      A, [=](auto MARKER) { MARKER[0] = -1; },
      Access::write(Sym<INT>("MARKER")));
  pl_reset->execute();

  auto pl = particle_loop(
      aa,
      [=](auto MARKER, auto index) {
        MARKER[0] = index.get_local_linear_index();
      },
      Access::write(Sym<INT>("MARKER")), Access::read(ParticleLoopIndex{}));
  pl->execute();

  INT index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      const INT ix = (*id)[0][rowx];
      if (ix % 2 == 0) {
        ASSERT_EQ(mx, index);
      } else {
        ASSERT_EQ(mx, -1);
      }
      index++;
    }
  }

  std::vector<INT> gav = {0};
  auto ga = std::make_shared<LocalArray<INT>>(sycl_target, gav);
  pl = particle_loop(
      aa,
      [=](auto MARKER, auto index, auto GA) {
        MARKER[0] = index.get_loop_linear_index();
        GA.fetch_add(0, 1);
      },
      Access::write(Sym<INT>("MARKER")), Access::read(ParticleLoopIndex{}),
      Access::add(ga));

  pl_reset->execute();
  pl->execute();
  gav = ga->get();
  const int npart_la = gav.at(0);

  std::set<INT> found_indices;
  const INT npart = aa->get_npart_local();
  index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      const INT ix = (*id)[0][rowx];
      if (ix % 2 == 0) {
        ASSERT_TRUE(mx < npart);
        ASSERT_TRUE(mx > -1);
        found_indices.insert(mx);
        index++;
      } else {
        ASSERT_EQ(mx, -1);
      }
    }
  }
  ASSERT_EQ(npart_la, npart);
  ASSERT_EQ(index, npart);
  ASSERT_EQ(found_indices.size(), npart);

  auto loop_indexing = particle_loop(
      aa,
      [](auto index, auto MARKER) {
        MARKER.at(0) = index.get_loop_linear_index();
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("MARKER")));

  for (int cx = 0; cx < cell_count; cx++) {
    loop_indexing->execute(cx);
  }

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    index = 0;
    found_indices.clear();

    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      const INT ix = (*id)[0][rowx];
      if (ix % 2 == 0) {
        ASSERT_TRUE(mx < nrow);
        ASSERT_TRUE(mx > -1);
        found_indices.insert(mx);
        index++;
      } else {
        ASSERT_EQ(mx, -1);
      }
    }
    ASSERT_EQ(found_indices.size(), index);
    ASSERT_EQ(found_indices.size(), aa->get_npart_cell(cellx));
  }

  auto loop_sub_indexing = particle_loop(
      aa,
      [](auto index, auto MARKER) {
        MARKER.at(0) = index.get_sub_linear_index();
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("MARKER")));
  for (int cx = 0; cx < cell_count; cx++) {
    loop_sub_indexing->execute(cx);
  }

  std::set<INT> sub_indices, sub_correct;
  const int npart_local = aa->get_npart_local();
  for (int ix = 0; ix < npart_local; ix++) {
    sub_correct.insert(ix);
  }

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_cell(Sym<INT>("MARKER"), cellx);
    auto id = A->get_cell(Sym<INT>("ID"), cellx);
    const int nrow = marker->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      if (id->at(rowx, 0) % 2 == 0) {
        const INT mx = marker->at(rowx, 0);
        sub_indices.insert(mx);
      }
    }
  }
  ASSERT_EQ(sub_correct, sub_indices);

  loop_sub_indexing->execute();

  sub_indices.clear();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_cell(Sym<INT>("MARKER"), cellx);
    auto id = A->get_cell(Sym<INT>("ID"), cellx);
    const int nrow = marker->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      if (id->at(rowx, 0) % 2 == 0) {
        const INT mx = marker->at(rowx, 0);
        sub_indices.insert(mx);
      }
    }
  }
  ASSERT_EQ(sub_correct, sub_indices);

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, whole_group) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto pl_set = particle_loop(
      A, [](auto m) { m.at(0) = 2; }, Access::write(Sym<INT>("MARKER")));
  pl_set->execute();

  auto aa = std::make_shared<ParticleSubGroup>(A);
  ASSERT_TRUE(aa->is_entire_particle_group());

  auto pl_set_aa = particle_loop(
      aa, [](auto m) { m.at(0) -= 1; }, Access::write(Sym<INT>("MARKER")));
  pl_set_aa->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      ASSERT_EQ(mx, 1);
    }
  }

  ASSERT_EQ(A->get_npart_local(), aa->get_npart_local());

  A->remove_particles(aa);
  ASSERT_EQ(A->get_npart_local(), 0);
  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(A->get_npart_cell(cx), 0);
    ASSERT_EQ(aa->get_npart_cell(cx), 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
