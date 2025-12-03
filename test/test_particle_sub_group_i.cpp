#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, truncate_selector) {

  const int npart_cell = 211;
  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, 2);

  auto to_remove = particle_sub_group(A, cell_count - 1);
  A->remove_particles(to_remove);

  {
    const int num_particles = 127;
    auto selector = std::make_shared<
        ParticleSubGroupImplementation::TruncateSubGroupSelector>(
        A, num_particles);

    ParticleSubGroupImplementation::Selection s;
    selector->create(&s);

    int npart = 0;
    std::vector<int> h_npart_cell(cell_count);
    std::vector<INT> h_npart_cell_es(cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = A->get_npart_cell(cellx);
      const int npart_cell_new = std::min(num_particles, npart_cell);
      h_npart_cell.at(cellx) = npart_cell_new;
      h_npart_cell_es.at(cellx) = npart;
      npart += npart_cell_new;
    }

    ASSERT_EQ(s.npart_local, npart);
    ASSERT_EQ(s.ncell, cell_count);

    std::vector<int> h_npart_cell_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.d_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.h_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);

    std::vector<INT> h_npart_cell_es_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_es_to_test.data(), s.d_npart_cell_es,
                cell_count * sizeof(INT))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_es_to_test, h_npart_cell_es);

    auto h_map =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s);

    // The whole particle group case should have an ordered map
    std::vector<INT> map;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = A->get_npart_cell(cellx);
      const int npart_cell_new = std::min(num_particles, npart_cell);
      map.resize(npart_cell_new);
      std::iota(map.begin(), map.end(), 0);
      ASSERT_EQ(map, h_map.at(cellx));
      ASSERT_TRUE(h_map.at(cellx).size() <= num_particles);
    }

    auto bb = particle_sub_group_truncate(A, 21);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = bb->get_npart_cell(cellx);
      ASSERT_TRUE(npart_cell <= 21);
    }

    auto AA = particle_sub_group(A);
    auto cc = particle_sub_group_truncate(AA, num_particles, true);

    auto s2 = cc->get_selection();
    auto h_map2 =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s2);
    ASSERT_EQ(h_map, h_map2);
  }

  {

    auto aa = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2; }, Access::read(Sym<INT>("ID")));

    const int num_particles = 2;
    auto selector = std::make_shared<
        ParticleSubGroupImplementation::TruncateSubGroupSelector>(
        aa, num_particles);

    ParticleSubGroupImplementation::Selection s;
    selector->create(&s);

    int npart = 0;
    std::vector<int> h_npart_cell(cell_count);
    std::vector<INT> h_npart_cell_es(cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = aa->get_npart_cell(cellx);
      const int npart_cell_new = std::min(num_particles, npart_cell);
      h_npart_cell.at(cellx) = npart_cell_new;
      h_npart_cell_es.at(cellx) = npart;
      npart += npart_cell_new;
    }

    ASSERT_EQ(s.npart_local, npart);
    ASSERT_EQ(s.ncell, cell_count);

    std::vector<int> h_npart_cell_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.d_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.h_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);

    std::vector<INT> h_npart_cell_es_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_es_to_test.data(), s.d_npart_cell_es,
                cell_count * sizeof(INT))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_es_to_test, h_npart_cell_es);

    auto h_map =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s);

    auto s_aa = aa->get_selection();
    auto h_map_aa =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s_aa);

    for (int cellx = 0; cellx < cell_count; cellx++) {
      std::set<INT> correct_layers;
      const int nrow =
          std::min(num_particles, static_cast<int>(h_map_aa[cellx].size()));
      for (int rowx = 0; rowx < nrow; rowx++) {
        correct_layers.insert(h_map_aa[cellx][rowx]);
      }

      std::set<INT> to_test_layers;
      for (auto ix : h_map.at(cellx)) {
        to_test_layers.insert(ix);
      }
      ASSERT_EQ(to_test_layers, correct_layers);
      ASSERT_TRUE(h_map.at(cellx).size() <= num_particles);
    }
    auto bb = particle_sub_group_truncate(A, 21);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = bb->get_npart_cell(cellx);
      ASSERT_TRUE(npart_cell <= 21);
    }

    auto AA = particle_sub_group(aa);
    auto cc = particle_sub_group_truncate(AA, num_particles, true);

    auto s2 = cc->get_selection();
    auto h_map2 =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s2);

    for (int cellx = 0; cellx < cell_count; cellx++) {
      std::set<INT> correct;
      for (auto ix : h_map.at(cellx)) {
        correct.insert(ix);
      }
      std::set<INT> to_test;
      for (auto ix : h_map2.at(cellx)) {
        to_test.insert(ix);
      }
      ASSERT_EQ(to_test, correct);
    }
  }

  {
    A->clear();
    auto aa = particle_sub_group_truncate(A, 2, true);
    auto s = aa->get_selection();
    ASSERT_EQ(s.ncell, cell_count);
    ASSERT_EQ(s.npart_local, 0);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      ASSERT_EQ(s.h_npart_cell[cellx], 0);
    }
  }

  A->free();
  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticleSubGroup, discard_selector) {

  const int npart_cell = 211;
  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, 2);

  auto to_remove = particle_sub_group(A, cell_count - 1);
  A->remove_particles(to_remove);

  {
    const int num_particles = 127;
    auto selector = std::make_shared<
        ParticleSubGroupImplementation::DiscardSubGroupSelector>(A,
                                                                 num_particles);

    ParticleSubGroupImplementation::Selection s;
    selector->create(&s);

    int npart = 0;
    std::vector<int> h_npart_cell(cell_count);
    std::vector<INT> h_npart_cell_es(cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = A->get_npart_cell(cellx);
      const int npart_cell_new = std::max(0, npart_cell - num_particles);
      h_npart_cell.at(cellx) = npart_cell_new;
      h_npart_cell_es.at(cellx) = npart;
      npart += npart_cell_new;
    }

    ASSERT_EQ(s.npart_local, npart);
    ASSERT_EQ(s.ncell, cell_count);

    std::vector<int> h_npart_cell_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.d_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.h_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);

    std::vector<INT> h_npart_cell_es_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_es_to_test.data(), s.d_npart_cell_es,
                cell_count * sizeof(INT))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_es_to_test, h_npart_cell_es);

    auto h_map =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s);

    // The whole particle group case should have an ordered map
    std::vector<INT> map;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = A->get_npart_cell(cellx);
      const int npart_cell_new = std::max(0, npart_cell - num_particles);
      map.resize(npart_cell_new);
      std::iota(map.begin(), map.end(), num_particles);
      ASSERT_EQ(map, h_map.at(cellx));
    }

    auto AA = particle_sub_group(A);
    auto cc = particle_sub_group_discard(AA, num_particles, true);

    auto s2 = cc->get_selection();
    auto h_map2 =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s2);
    ASSERT_EQ(h_map, h_map2);
  }

  {

    auto aa = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2; }, Access::read(Sym<INT>("ID")));

    const int num_particles = 2;
    auto selector = std::make_shared<
        ParticleSubGroupImplementation::DiscardSubGroupSelector>(aa,
                                                                 num_particles);

    ParticleSubGroupImplementation::Selection s;
    selector->create(&s);

    int npart = 0;
    std::vector<int> h_npart_cell(cell_count);
    std::vector<INT> h_npart_cell_es(cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = aa->get_npart_cell(cellx);
      const int npart_cell_new = std::max(0, npart_cell - num_particles);
      h_npart_cell.at(cellx) = npart_cell_new;
      h_npart_cell_es.at(cellx) = npart;
      npart += npart_cell_new;
    }

    ASSERT_EQ(s.npart_local, npart);
    ASSERT_EQ(s.ncell, cell_count);

    std::vector<int> h_npart_cell_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.d_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);
    sycl_target->queue
        .memcpy(h_npart_cell_to_test.data(), s.h_npart_cell,
                cell_count * sizeof(int))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_to_test, h_npart_cell);

    std::vector<INT> h_npart_cell_es_to_test(cell_count);
    sycl_target->queue
        .memcpy(h_npart_cell_es_to_test.data(), s.d_npart_cell_es,
                cell_count * sizeof(INT))
        .wait_and_throw();
    ASSERT_EQ(h_npart_cell_es_to_test, h_npart_cell_es);

    auto h_map =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s);

    auto s_aa = aa->get_selection();
    auto h_map_aa =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s_aa);

    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int nrow =
          std::max(0, static_cast<int>(h_map_aa[cellx].size()) - num_particles);

      std::set<INT> correct_layers;
      for (int rowx = 0; rowx < nrow; rowx++) {
        correct_layers.insert(h_map_aa[cellx][rowx + num_particles]);
      }

      std::set<INT> to_test_layers;
      for (auto ix : h_map.at(cellx)) {
        to_test_layers.insert(ix);
      }
      ASSERT_EQ(to_test_layers, correct_layers);
    }

    auto AA = particle_sub_group(aa);
    auto cc = particle_sub_group_discard(AA, num_particles, true);

    auto s2 = cc->get_selection();
    auto h_map2 =
        ParticleSubGroupImplementation::get_host_map_cells_to_particles(
            sycl_target, s2);

    for (int cellx = 0; cellx < cell_count; cellx++) {
      std::set<INT> correct;
      for (auto ix : h_map.at(cellx)) {
        correct.insert(ix);
      }
      std::set<INT> to_test;
      for (auto ix : h_map2.at(cellx)) {
        to_test.insert(ix);
      }
      ASSERT_EQ(to_test, correct);
    }
  }

  {
    A->clear();
    auto aa = particle_sub_group_discard(A, 2, true);
    auto s = aa->get_selection();
    ASSERT_EQ(s.npart_local, 0);
    ASSERT_EQ(s.ncell, cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      ASSERT_EQ(s.h_npart_cell[cellx], 0);
    }
  }

  A->free();
  sycl_target->free();
  A->domain->mesh->free();
}

TEST(ParticleSubGroup, disjoint_union) {

  const int npart_cell = 211;
  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, 2);

  {
    const int npart_local_before = A->get_npart_local();
    const int cell_to_empty = std::max(0, cell_count - 3);
    const int npart_to_remove = A->get_npart_cell(cell_to_empty);
    auto to_remove = particle_sub_group(A, cell_to_empty);
    A->remove_particles(to_remove);
    ASSERT_EQ(A->get_npart_cell(cell_to_empty), 0);
    ASSERT_EQ(A->get_npart_local(), npart_local_before - npart_to_remove);
  }

  {

    std::vector<ParticleSubGroupSharedPtr> groups = {particle_sub_group(A)};
    auto aa = particle_sub_group_disjoint_union(groups);

    ASSERT_NE(
        std::dynamic_pointer_cast<ParticleSubGroupImplementation::CopySelector>(
            aa->selector),
        nullptr);
  }

  {

    std::vector<ParticleSubGroupSharedPtr> groups = {
        particle_sub_group(particle_sub_group(
            A, [=](auto ID) { return ID.at(0) % 2; },
            Access::read(Sym<INT>("ID"))))};
    auto aa = particle_sub_group_disjoint_union(groups);

    ASSERT_NE(
        std::dynamic_pointer_cast<ParticleSubGroupImplementation::CopySelector>(
            aa->selector),
        nullptr);
  }

  A->add_particle_dat(Sym<INT>("ID2"), 1);
  particle_loop(
      A, [=](auto ID, auto ID2) { ID2.at(0) = ID.at(0); },
      Access::read(Sym<INT>("ID")), Access::write(Sym<INT>("ID2")))
      ->execute();

  {
    std::vector<ParticleSubGroupSharedPtr> groups;
    for (int ix = 0; ix < 5; ix++) {
      if (ix % 2 == 0) {
        groups.push_back(particle_sub_group(
            A, [=](auto ID) { return ID.at(0) % 5 == ix; },
            Access::read(Sym<INT>("ID"))));
      } else {
        groups.push_back(particle_sub_group(
            A, [=](auto ID) { return ID.at(0) % 5 == ix; },
            Access::read(Sym<INT>("ID2"))));
      }
    }

    auto aa = particle_sub_group_disjoint_union(groups);
    ASSERT_EQ(aa->get_npart_local(), A->get_npart_local());

    {
      ASSERT_FALSE(aa->create_if_required());
      auto d_ID = Access::direct_get(Access::write(A->get_dat(Sym<INT>("ID"))));
      Access::direct_restore(Access::write(A->get_dat(Sym<INT>("ID"))), d_ID);
      ASSERT_TRUE(aa->create_if_required());
      ASSERT_FALSE(aa->create_if_required());
      auto d_ID2 =
          Access::direct_get(Access::write(A->get_dat(Sym<INT>("ID2"))));
      Access::direct_restore(Access::write(A->get_dat(Sym<INT>("ID2"))), d_ID2);
      ASSERT_TRUE(aa->create_if_required());
      ASSERT_FALSE(aa->create_if_required());
    }

    {

      int npart_local = 0;

      std::vector<int> t_npart_cell(cell_count);
      std::vector<int> h_npart_cell(cell_count);
      std::vector<INT> t_npart_cell_es(cell_count);
      std::vector<INT> h_npart_cell_es(cell_count);
      std::fill(h_npart_cell.begin(), h_npart_cell.end(), 0);
      std::fill(h_npart_cell_es.begin(), h_npart_cell_es.end(), 0);

      // Get the original maps
      std::vector<std::vector<std::vector<INT>>> h_maps_correct;
      for (int ix = 0; ix < 5; ix++) {
        groups.at(ix)->create_if_required();
        auto selection = groups.at(ix)->get_selection();
        npart_local += selection.npart_local;
        ASSERT_EQ(selection.ncell, cell_count);
        h_maps_correct.push_back(
            get_host_map_cells_to_particles(sycl_target, selection));

        sycl_target->queue
            .memcpy(t_npart_cell_es.data(), selection.d_npart_cell_es,
                    cell_count * sizeof(INT))
            .wait_and_throw();

        for (int cellx = 0; cellx < cell_count; cellx++) {
          h_npart_cell.at(cellx) += selection.h_npart_cell[cellx];
          h_npart_cell_es.at(cellx) += t_npart_cell_es[cellx];
        }
      }

      std::vector<std::set<INT>> h_map_correct_set(cell_count);
      for (int ix = 0; ix < 5; ix++) {
        for (int cellx = 0; cellx < cell_count; cellx++) {
          for (auto &l : h_maps_correct[ix][cellx]) {
            h_map_correct_set[cellx].insert(l);
          }
        }
      }

      // get the created maps
      aa->create_if_required();
      auto selection_to_test = aa->get_selection();

      sycl_target->queue
          .memcpy(t_npart_cell_es.data(), selection_to_test.d_npart_cell_es,
                  cell_count * sizeof(INT))
          .wait_and_throw();

      sycl_target->queue
          .memcpy(t_npart_cell.data(), selection_to_test.d_npart_cell,
                  cell_count * sizeof(int))
          .wait_and_throw();

      for (int cellx = 0; cellx < cell_count; cellx++) {
        ASSERT_EQ(h_npart_cell.at(cellx),
                  selection_to_test.h_npart_cell[cellx]);
        ASSERT_EQ(h_npart_cell.at(cellx), t_npart_cell.at(cellx));
        ASSERT_EQ(h_npart_cell_es.at(cellx), t_npart_cell_es.at(cellx));
      }

      std::vector<std::vector<INT>> h_map_to_test =
          get_host_map_cells_to_particles(sycl_target, selection_to_test);

      ASSERT_EQ(selection_to_test.npart_local, npart_local);
      ASSERT_EQ(selection_to_test.ncell, cell_count);

      std::vector<std::set<INT>> h_map_to_test_set(cell_count);
      for (int cellx = 0; cellx < cell_count; cellx++) {
        for (auto &l : h_map_to_test[cellx]) {
          h_map_to_test_set[cellx].insert(l);
        }
      }

      for (int cellx = 0; cellx < cell_count; cellx++) {
        ASSERT_EQ(h_map_to_test_set[cellx], h_map_correct_set[cellx]);
      }
    }

    A->clear();
    ASSERT_TRUE(aa->create_if_required());
    ASSERT_FALSE(aa->create_if_required());
  }

  A->free();
  sycl_target->free();
  A->domain->mesh->free();
}
