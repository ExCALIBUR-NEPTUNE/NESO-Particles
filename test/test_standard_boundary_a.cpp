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

namespace {

struct NormalInformation {
  INT element_id{0};
  REAL normal[3]{0.0, 0.0, 0.0};
  REAL bounding_box[6]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

inline bool contains_point(const NormalInformation *normal_info, const REAL p0,
                           const REAL p1, const REAL p2, const REAL tol) {

  const auto bb = normal_info->bounding_box;
  const bool contained = (((bb[0] - tol) <= p0) && (p0 <= (bb[3 + 0] + tol))) &&
                         (((bb[1] - tol) <= p1) && (p1 <= (bb[3 + 1] + tol))) &&
                         (((bb[2] - tol) <= p2) && (p2 <= (bb[3 + 2] + tol)));

  return contained;
}

} // namespace

TEST(CartesianTrajectoryIntersection, offsets_2d) {

  const int ncell_x = 16;
  const int ncell_y = 32;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_2d(27, ncell_x, ncell_y);

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

  const int num_facets = 2 * ncell_x + 2 * ncell_y;
  auto correct_lut = std::make_shared<LookupTable<INT, NormalInformation>>(
      sycl_target, num_facets);

  int index = 0;
  for (int fx = 0; fx < ncell_x; fx++) {
    NormalInformation normal_info;
    normal_info.element_id = index;
    normal_info.normal[1] = 1.0;
    // These cells have width 1.
    normal_info.bounding_box[0] = fx;
    normal_info.bounding_box[3 + 0] = fx + 1.0;
    correct_lut->add(index, normal_info);
    index++;
  }
  for (int fx = 0; fx < ncell_y; fx++) {
    NormalInformation normal_info;
    normal_info.element_id = index;
    normal_info.normal[0] = -1.0;
    normal_info.bounding_box[0] = ncell_x;
    normal_info.bounding_box[3 + 0] = ncell_x;
    normal_info.bounding_box[1] = fx;
    normal_info.bounding_box[3 + 1] = fx + 1.0;
    correct_lut->add(index, normal_info);
    index++;
  }
  for (int fx = 0; fx < ncell_x; fx++) {
    NormalInformation normal_info;
    normal_info.element_id = index;
    normal_info.normal[1] = -1.0;
    normal_info.bounding_box[0] = fx;
    normal_info.bounding_box[3 + 0] = fx + 1.0;
    normal_info.bounding_box[1] = ncell_y;
    normal_info.bounding_box[3 + 1] = ncell_y;
    correct_lut->add(index, normal_info);
    index++;
  }
  for (int fx = 0; fx < ncell_y; fx++) {
    NormalInformation normal_info;
    normal_info.element_id = index;
    normal_info.normal[0] = 1.0;
    normal_info.bounding_box[1] = fx;
    normal_info.bounding_box[3 + 1] = fx + 1.0;
    correct_lut->add(index, normal_info);
    index++;
  }
  ASSERT_EQ(index, num_facets);

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

  auto lambda_test = [&](auto iteration_set, auto offsetx, auto offsety,
                         const int modified_index,
                         const REAL correct_truncation,
                         const int unmodified_index) {
    particle_loop(
        iteration_set,
        [=](auto P, auto P_ORIG) {
          P.at(0) = P_ORIG.at(0);
          P.at(1) = P_ORIG.at(1);
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("P_ORIG")))
        ->execute();

    cartesian_trajectory_intersection->pre_integration(iteration_set);

    particle_loop(
        iteration_set,
        [=](auto P) {
          P.at(0) += offsetx;
          P.at(1) += offsety;
        },
        Access::write(Sym<REAL>("P")))
        ->execute();

    auto groups =
        cartesian_trajectory_intersection->post_integration(iteration_set);

    for (int boundaryx : {0, 1}) {
      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      auto k_correct_lut = correct_lut->root;
      particle_loop(
          groups.at(boundaryx),
          [=](auto P, auto INTERSECTION_POINT, auto INTERSECTION_NORMAL,
              auto INTERSECTION_METADATA) {
            NESO_KERNEL_ASSERT(
                INTERSECTION_METADATA.at_ephemeral(0) % 2 == boundaryx, k_ep);

            const INT element_id = INTERSECTION_METADATA.at_ephemeral(1);
            const NormalInformation *normal_info = nullptr;
            k_correct_lut->get(element_id, &normal_info);
            NESO_KERNEL_ASSERT(element_id == normal_info->element_id, k_ep);
            NESO_KERNEL_ASSERT(INTERSECTION_NORMAL.at_ephemeral(0) ==
                                   normal_info->normal[0],
                               k_ep);
            NESO_KERNEL_ASSERT(INTERSECTION_NORMAL.at_ephemeral(1) ==
                                   normal_info->normal[1],
                               k_ep);
            NESO_KERNEL_ASSERT(
                Kernel::abs(INTERSECTION_POINT.at_ephemeral(modified_index) -
                            correct_truncation) < 1.0e-12,
                k_ep);
            NESO_KERNEL_ASSERT(
                Kernel::abs(INTERSECTION_POINT.at_ephemeral(unmodified_index) -
                            P.at(unmodified_index)) < 1.0e-12,
                k_ep);
            NESO_KERNEL_ASSERT(
                contains_point(normal_info, INTERSECTION_POINT.at_ephemeral(0),
                               INTERSECTION_POINT.at_ephemeral(1), 0.0,
                               1.0e-12),
                k_ep);
          },
          Access::read(Sym<REAL>("P")),
          Access::read(BoundaryInteractionSpecification::intersection_point),
          Access::read(BoundaryInteractionSpecification::intersection_normal),
          Access::read(BoundaryInteractionSpecification::intersection_metadata))
          ->execute();

      ASSERT_FALSE(ep.get_flag());
    }
  };

  lambda_test(A, 100.0, 0, 0, ncell_x, 1);
  lambda_test(A, -100.0, 0, 0, 0.0, 1);
  lambda_test(A, 0.0, 100.0, 1, ncell_y, 0);
  lambda_test(A, 0.0, -100.0, 1, 0.0, 0);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  lambda_test(aa, 100.0, 0, 0, ncell_x, 1);
  lambda_test(aa, -100.0, 0, 0, 0.0, 1);
  lambda_test(aa, 0.0, 100.0, 1, ncell_y, 0);
  lambda_test(aa, 0.0, -100.0, 1, 0.0, 0);

  sycl_target->free();
}
