#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, copy_selector) {
  auto A = subgroup_test_common();
  auto mesh = A->domain->mesh;
  auto sycl_target = A->sycl_target;

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
  ASSERT_EQ(selection_aa.h_npart_cell, selection_bb.h_npart_cell);
  ASSERT_EQ(selection_aa.d_npart_cell, selection_bb.d_npart_cell);
  ASSERT_EQ(selection_aa.d_npart_cell_es, selection_bb.d_npart_cell_es);
  ASSERT_EQ(selection_aa.d_map_cells_to_particles.map_ptr,
            selection_bb.d_map_cells_to_particles.map_ptr);

  particle_loop(
      A, [=](auto ID) { ID.at(0) += 3; }, Access::write(Sym<INT>("ID")))
      ->execute();

  ASSERT_TRUE(bb->create_if_required());
  ASSERT_FALSE(bb->create_if_required());
  ASSERT_FALSE(aa->create_if_required());

  sycl_target->free();
  mesh->free();
}
