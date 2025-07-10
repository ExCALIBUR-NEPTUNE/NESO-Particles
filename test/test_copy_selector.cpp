#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, copy_selector) {
  auto A = subgroup_test_common();
  auto mesh = A->domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 3 == 0; },
      Access::read(Sym<INT>("ID")));

  auto bb = particle_sub_group(aa);
  aa->create_if_required();
  ASSERT_TRUE(bb->create_if_required());
  ASSERT_FALSE(bb->create_if_required());

  auto selection_aa = aa->get_selection();
  auto selection_bb = bb->get_selection();

  ASSERT_EQ(selection_aa.npart_local, selection_bb.npart_local);
  ASSERT_EQ(selection_aa.ncell, selection_bb.ncell);

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(selection_aa.h_npart_cell[cx], selection_bb.h_npart_cell[cx]);
  }

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  auto k_npart_cell_aa = selection_aa.d_npart_cell;
  auto k_npart_cell_bb = selection_bb.d_npart_cell;
  auto k_npart_cell_es_aa = selection_aa.d_npart_cell_es;
  auto k_npart_cell_es_bb = selection_bb.d_npart_cell_es;

  sycl_target->queue
      .parallel_for(
          sycl::range<1>(cell_count),
          [=](auto cx) {
            NESO_KERNEL_ASSERT(k_npart_cell_aa[cx] == k_npart_cell_bb[cx],
                               k_ep);
            NESO_KERNEL_ASSERT(k_npart_cell_es_aa[cx] == k_npart_cell_es_bb[cx],
                               k_ep);
          })
      .wait_and_throw();
  ASSERT_FALSE(ep.get_flag());

  auto map_aa = get_host_map_cells_to_particles(sycl_target, selection_aa);
  auto map_bb = get_host_map_cells_to_particles(sycl_target, selection_bb);

  for (int cx = 0; cx < cell_count; cx++) {
    std::set<INT> set_aa;
    std::set<INT> set_bb;
    for (auto ix : map_aa.at(cx)) {
      set_aa.insert(ix);
    }
    for (auto ix : map_bb.at(cx)) {
      set_bb.insert(ix);
    }
    ASSERT_EQ(set_aa, set_bb);
  }

  particle_loop(
      A, [=](auto ID) { ID.at(0) += 3; }, Access::write(Sym<INT>("ID")))
      ->execute();

  ASSERT_TRUE(bb->create_if_required());
  ASSERT_FALSE(bb->create_if_required());
  ASSERT_FALSE(aa->create_if_required());

  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, copy_selector_static) {
  auto A = subgroup_test_common();
  auto mesh = A->domain->mesh;
  auto sycl_target = A->sycl_target;

  A->add_particle_dat(Sym<INT>("ID2"), 1);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  particle_loop(
      A, [=](auto ID, auto ID2) { ID2.at(0) = ID.at(0); },
      Access::read(Sym<INT>("ID")), Access::write(Sym<INT>("ID2")))
      ->execute();

  auto bb = static_particle_sub_group(aa);

  particle_loop(
      aa, [=](auto ID) { ID.at(0)--; }, Access::write(Sym<INT>("ID")))
      ->execute();

  ASSERT_FALSE(bb->create_if_required());
  ASSERT_TRUE(aa->create_if_required());
  ASSERT_FALSE(aa->create_if_required());

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      bb,
      [=](auto ID, auto ID2) {
        NESO_KERNEL_ASSERT(ID.at(0) % 2 != 0, k_ep);
        NESO_KERNEL_ASSERT(ID2.at(0) % 2 == 0, k_ep);
      },
      Access::read(Sym<INT>("ID")), Access::read(Sym<INT>("ID2")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
  mesh->free();
}
