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

namespace {}

TEST(CartesianTrajectoryIntersection, base_2d) {
  auto [A_t, sycl_target_t, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto sycl_target = sycl_target_t;
  auto A = A_t;

  const int ndim = 2;
  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0, 2};
  boundary_groups[1] = {1, 3};

  auto cartesian_trajectory_intersection =
      std::make_shared<CartesianTrajectoryIntersection>(
          sycl_target,
          std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
          boundary_groups);
  cartesian_trajectory_intersection->prepare_particle_group(A);

  A->add_particle_dat(Sym<REAL>("P_ORIG"), ndim);

  particle_loop(
      A,
      [=](auto P, auto P_ORIG) {
        for (int dx = 0; dx < ndim; dx++) {
          P_ORIG.at(dx) = P.at(dx);
        }
      },
      Access::read(Sym<REAL>("P")), Access::write(Sym<REAL>("P_ORIG")))
      ->execute();

  auto lambda_test = [&](auto offsetx, auto offsety) {
    cartesian_trajectory_intersection->pre_integration(A);

    particle_loop(
        A,
        [=](auto P) {
          P.at(0) += offsetx;
          P.at(1) += offsety;
        },
        Access::write(Sym<REAL>("P")))
        ->execute();

    auto groups = cartesian_trajectory_intersection->post_integration(A);

    for (int boundaryx : {0, 1}) {
      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      particle_loop(
          groups.at(boundaryx),
          [=](auto INTERSECTION_POINT, auto INTERSECTION_NORMAL,
              auto INTERSECTION_METADATA) {
            NESO_KERNEL_ASSERT(INTERSECTION_METADATA.at(0) % 2 == boundaryx,
                               k_ep);
          },
          Access::read(BoundaryInteractionSpecification::intersection_point),
          Access::read(BoundaryInteractionSpecification::intersection_normal),
          Access::read(BoundaryInteractionSpecification::intersection_metadata))
          ->execute();

      ASSERT_FALSE(ep.get_flag());
    }
  };

  lambda_test(100.0, 0);

  sycl_target->free();
}
