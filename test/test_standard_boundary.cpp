#include "include/test_neso_particles.hpp"

TEST(StandardBoundary, base) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  ASSERT_FALSE(contains_boundary_interaction_data(aa));
  ASSERT_FALSE(contains_boundary_interaction_data(aa, 2));

  aa->add_ephemeral_dat(BoundaryInteractionSpecification::intersection_normal,
                        2);
  aa->add_ephemeral_dat(BoundaryInteractionSpecification::intersection_point,
                        2);
  aa->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_metadata,
      BoundaryInteractionSpecification::intersection_metadata_ncomp);

  ASSERT_TRUE(contains_boundary_interaction_data(aa));
  ASSERT_TRUE(contains_boundary_interaction_data(aa, 2));
  ASSERT_FALSE(contains_boundary_interaction_data(aa, 3));

  sycl_target->free();
}
