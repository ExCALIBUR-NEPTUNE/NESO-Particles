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

    auto bb = particle_sub_group_truncation(A, 21);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = bb->get_npart_cell(cellx);
      ASSERT_TRUE(npart_cell <= 21);
    }
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

    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto ID = A->get_cell(Sym<INT>("ID"), cellx);
      const int nrow = ID->nrow;

      std::set<INT> correct_layers;
      int counter = 0;
      for (int rowx = 0; rowx < nrow; rowx++) {
        if (ID->at(rowx, 0) % 2) {
          if (counter < num_particles) {
            correct_layers.insert(rowx);
          }
          counter++;
        }
      }

      std::set<INT> to_test_layers;
      for (auto ix : h_map.at(cellx)) {
        to_test_layers.insert(ix);
      }
      ASSERT_EQ(to_test_layers, correct_layers);
      ASSERT_TRUE(h_map.at(cellx).size() <= num_particles);
    }
    auto bb = particle_sub_group_truncation(A, 21);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int npart_cell = bb->get_npart_cell(cellx);
      ASSERT_TRUE(npart_cell <= 21);
    }
  }

  A->free();
  sycl_target->free();
  A->domain->mesh->free();
}
