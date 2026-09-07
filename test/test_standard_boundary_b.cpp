#include "include/test_neso_particles.hpp"

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

TEST(CartesianTrajectoryIntersection, offsets_3d) {

  const int ncell_x = 12;
  const int ncell_y = 7;
  const int ncell_z = 8;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_3d(4, ncell_x, ncell_y, ncell_z);

  auto sycl_target = sycl_target_t;
  auto A = A_t;

  const int ndim = 3;
  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0, 2};
  boundary_groups[1] = {1, 3};
  boundary_groups[2] = {4, 5};

  auto cartesian_trajectory_intersection =
      std::make_shared<CartesianTrajectoryIntersection>(
          sycl_target,
          std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
          boundary_groups);
  cartesian_trajectory_intersection->prepare_particle_group(A);

  const int num_facets =
      2 * ncell_x * ncell_y + 2 * ncell_x * ncell_z + 2 * ncell_y * ncell_z;
  auto correct_lut = std::make_shared<LookupTable<INT, NormalInformation>>(
      sycl_target, num_facets);

  INT index = 0;

  for (int bx = 0; bx < ncell_z; bx++) {
    for (int ax = 0; ax < ncell_x; ax++) {
      NormalInformation normal_info;
      normal_info.element_id = index;
      normal_info.normal[1] = 1.0;
      // These cells have width 1.
      normal_info.bounding_box[0] = ax;
      normal_info.bounding_box[3 + 0] = ax + 1.0;
      normal_info.bounding_box[2] = bx;
      normal_info.bounding_box[3 + 2] = bx + 1.0;
      correct_lut->add(index, normal_info);
      index++;
    }
  }
  for (int bx = 0; bx < ncell_z; bx++) {
    for (int ax = 0; ax < ncell_y; ax++) {
      NormalInformation normal_info;
      normal_info.element_id = index;
      normal_info.normal[0] = -1.0;
      // These cells have width 1.
      normal_info.bounding_box[0] = ncell_x;
      normal_info.bounding_box[3 + 0] = ncell_x;
      normal_info.bounding_box[1] = ax;
      normal_info.bounding_box[3 + 1] = ax + 1.0;
      normal_info.bounding_box[2] = bx;
      normal_info.bounding_box[3 + 2] = bx + 1.0;
      correct_lut->add(index, normal_info);
      index++;
    }
  }
  for (int bx = 0; bx < ncell_z; bx++) {
    for (int ax = 0; ax < ncell_x; ax++) {
      NormalInformation normal_info;
      normal_info.element_id = index;
      normal_info.normal[1] = -1.0;
      // These cells have width 1.
      normal_info.bounding_box[0] = ax;
      normal_info.bounding_box[3 + 0] = ax + 1;
      normal_info.bounding_box[1] = ncell_y;
      normal_info.bounding_box[3 + 1] = ncell_y;
      normal_info.bounding_box[2] = bx;
      normal_info.bounding_box[3 + 2] = bx + 1.0;
      correct_lut->add(index, normal_info);
      index++;
    }
  }
  for (int bx = 0; bx < ncell_z; bx++) {
    for (int ax = 0; ax < ncell_y; ax++) {
      NormalInformation normal_info;
      normal_info.element_id = index;
      normal_info.normal[0] = 1.0;
      // These cells have width 1.
      normal_info.bounding_box[1] = ax;
      normal_info.bounding_box[3 + 1] = ax + 1.0;
      normal_info.bounding_box[2] = bx;
      normal_info.bounding_box[3 + 2] = bx + 1.0;
      correct_lut->add(index, normal_info);
      index++;
    }
  }

  for (int bx = 0; bx < ncell_y; bx++) {
    for (int ax = 0; ax < ncell_x; ax++) {
      NormalInformation normal_info;
      normal_info.element_id = index;
      normal_info.normal[2] = 1.0;
      // These cells have width 1.
      normal_info.bounding_box[0] = ax;
      normal_info.bounding_box[3 + 0] = ax + 1.0;
      normal_info.bounding_box[1] = bx;
      normal_info.bounding_box[3 + 1] = bx + 1.0;
      correct_lut->add(index, normal_info);
      index++;
    }
  }
  for (int bx = 0; bx < ncell_y; bx++) {
    for (int ax = 0; ax < ncell_x; ax++) {
      NormalInformation normal_info;
      normal_info.element_id = index;
      normal_info.normal[2] = -1.0;
      // These cells have width 1.
      normal_info.bounding_box[0] = ax;
      normal_info.bounding_box[3 + 0] = ax + 1.0;
      normal_info.bounding_box[1] = bx;
      normal_info.bounding_box[3 + 1] = bx + 1.0;
      normal_info.bounding_box[2] = ncell_z;
      normal_info.bounding_box[3 + 2] = ncell_z;
      correct_lut->add(index, normal_info);
      index++;
    }
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
                         auto offsetz, const int modified_index,
                         const REAL correct_truncation,
                         const int unmodified_index0,
                         const int unmodified_index1) {
    particle_loop(
        iteration_set,
        [=](auto P, auto P_ORIG) {
          P.at(0) = P_ORIG.at(0);
          P.at(1) = P_ORIG.at(1);
          P.at(2) = P_ORIG.at(2);
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("P_ORIG")))
        ->execute();

    cartesian_trajectory_intersection->pre_integration(iteration_set);

    particle_loop(
        iteration_set,
        [=](auto P) {
          P.at(0) += offsetx;
          P.at(1) += offsety;
          P.at(2) += offsetz;
        },
        Access::write(Sym<REAL>("P")))
        ->execute();

    auto groups =
        cartesian_trajectory_intersection->post_integration(iteration_set);

    for (int boundaryx : {0, 1, 2}) {
      ErrorPropagate ep(sycl_target);
      auto k_ep = ep.device_ptr();
      auto k_correct_lut = correct_lut->root;
      particle_loop(
          groups.at(boundaryx),
          [=](auto P, auto INTERSECTION_POINT, auto INTERSECTION_NORMAL,
              auto INTERSECTION_METADATA) {
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
            NESO_KERNEL_ASSERT(INTERSECTION_NORMAL.at_ephemeral(2) ==
                                   normal_info->normal[2],
                               k_ep);
            NESO_KERNEL_ASSERT(
                Kernel::abs(INTERSECTION_POINT.at_ephemeral(modified_index) -
                            correct_truncation) < 1.0e-12,
                k_ep);
            NESO_KERNEL_ASSERT(
                Kernel::abs(INTERSECTION_POINT.at_ephemeral(unmodified_index0) -
                            P.at(unmodified_index0)) < 1.0e-12,
                k_ep);
            NESO_KERNEL_ASSERT(
                Kernel::abs(INTERSECTION_POINT.at_ephemeral(unmodified_index1) -
                            P.at(unmodified_index1)) < 1.0e-12,
                k_ep);
            NESO_KERNEL_ASSERT(
                contains_point(normal_info, INTERSECTION_POINT.at_ephemeral(0),
                               INTERSECTION_POINT.at_ephemeral(1),
                               INTERSECTION_POINT.at_ephemeral(2), 1.0e-12),
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

  lambda_test(A, 100.0, 0.0, 0.0, 0, ncell_x, 1, 2);
  lambda_test(A, 0.0, 100.0, 0.0, 1, ncell_y, 0, 2);
  lambda_test(A, 0.0, 0.0, 100.0, 2, ncell_z, 0, 1);
  lambda_test(A, 0.0, -100.0, 0.0, 1, 0.0, 0, 2);
  lambda_test(A, -100.0, 0.0, 0.0, 0, 0.0, 1, 2);
  lambda_test(A, 0.0, -100.0, 0.0, 1, 0.0, 0, 2);
  lambda_test(A, 0.0, 0.0, -100.0, 2, 0.0, 0, 1);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  lambda_test(aa, 100.0, 0.0, 0.0, 0, ncell_x, 1, 2);
  lambda_test(aa, 0.0, 100.0, 0.0, 1, ncell_y, 0, 2);
  lambda_test(aa, 0.0, 0.0, 100.0, 2, ncell_z, 0, 1);
  lambda_test(aa, 0.0, -100.0, 0.0, 1, 0.0, 0, 2);
  lambda_test(aa, -100.0, 0.0, 0.0, 0, 0.0, 1, 2);
  lambda_test(aa, 0.0, -100.0, 0.0, 1, 0.0, 0, 2);
  lambda_test(aa, 0.0, 0.0, -100.0, 2, 0.0, 0, 1);

  cartesian_trajectory_intersection->free();
  sycl_target->free();
  A->domain->mesh->free();
}

TEST(CartesianTrajectoryIntersection, labels_3d) {

  const int ncell_x = 15;
  const int ncell_y = 14;
  const int ncell_z = 31;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_3d(4, ncell_x, ncell_y, ncell_z);
  auto sycl_target = sycl_target_t;
  auto A = A_t;

  {
    auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);
    INT bound_lower = 0;
    INT bound_upper = 0;
    mesh->get_global_face_index_bounds(bound_lower, bound_upper);
    std::set<INT> correct, to_test;
    for (INT ix = bound_lower; ix < bound_upper; ix++) {
      correct.insert(ix);
    }

    auto face_cells = mesh->get_owned_face_cells();
    for (INT fx : face_cells) {
      to_test.insert(fx);
    }

    auto to_test2 = set_all_reduce_union(to_test, mesh->get_comm());
    ASSERT_EQ(correct, to_test2);
  }

  {
    auto aa = particle_sub_group(
        A, [=](auto INDEX) { return INDEX.get_loop_linear_index() != 0; },
        Access::read(ParticleLoopIndex{}));

    A->remove_particles(aa);
    ASSERT_EQ(A->get_npart_local(), 1);
  }

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0};
  boundary_groups[1] = {1};
  boundary_groups[2] = {2};
  boundary_groups[3] = {3};
  boundary_groups[4] = {4};
  boundary_groups[5] = {5};
  auto cartesian_trajectory_intersection =
      std::make_shared<CartesianTrajectoryIntersection>(
          sycl_target,
          std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
          boundary_groups, 1.0e-10);
  cartesian_trajectory_intersection->prepare_particle_group(A);

  auto loop_reset = particle_loop(
      A,
      [=](auto P) {
        P.at(0) = 0.5;
        P.at(1) = 0.5;
        P.at(2) = 0.5;
      },
      Access::write(Sym<REAL>("P")));

  auto lambda_do_test = [&](const int expected_label, const REAL p0,
                            const REAL p1, const REAL p2) {
    loop_reset->execute();

    cartesian_trajectory_intersection->pre_integration(A);
    particle_loop(
        A,
        [=](auto P) {
          P.at(0) = p0;
          P.at(1) = p1;
          P.at(2) = p2;
        },
        Access::write(Sym<REAL>("P")))
        ->execute();

    auto groups = cartesian_trajectory_intersection->post_integration(A);

    for (auto &gx : groups) {
      const int correct = gx.first == expected_label ? 1 : 0;
      ASSERT_EQ(gx.second->get_npart_local(), correct);
    }
  };

  lambda_do_test(0, 0.5, -1000.0, 0.5);
  lambda_do_test(2, 0.5, 1000.0, 0.5);
  lambda_do_test(3, -1000.0, 0.5, 0.5);
  lambda_do_test(1, 1000.0, 0.5, 0.5);
  lambda_do_test(4, 0.5, 0.5, -1000.0);
  lambda_do_test(5, 0.5, 0.5, 1000.0);
}

TEST(CartesianTrajectoryIntersection, shallow_intersection_3d) {

  const int ncell_x = 15;
  const int ncell_y = 23;
  const int ncell_z = 16;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_3d(4, ncell_x, ncell_y, ncell_z);
  auto sycl_target = sycl_target_t;
  auto A = A_t;

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0};
  boundary_groups[1] = {1};
  boundary_groups[2] = {2};
  boundary_groups[3] = {3};
  boundary_groups[4] = {4};
  boundary_groups[5] = {5};

  auto cartesian_trajectory_intersection =
      std::make_shared<CartesianTrajectoryIntersection>(
          sycl_target,
          std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
          boundary_groups, 1.0e-10);
  cartesian_trajectory_intersection->prepare_particle_group(A);

  {
    auto aa = particle_sub_group(
        A, [=](auto INDEX) { return INDEX.get_loop_linear_index() != 0; },
        Access::read(ParticleLoopIndex{}));

    A->remove_particles(aa);
    ASSERT_EQ(A->get_npart_local(), 1);
  }

  const REAL offset_eps = 1.0e-12;
  particle_loop(
      A,
      [=](auto P) {
        P.at(0) = 0.5;
        P.at(1) = 0.5;
        P.at(2) = offset_eps;
      },
      Access::write(Sym<REAL>("P")))
      ->execute();

  cartesian_trajectory_intersection->pre_integration(A);

  particle_loop(
      A,
      [=](auto P) {
        P.at(0) = 0.5;
        // This y position has to be different to minus the previous y position
        // otherwise the trajectory does actually go through the corner.
        P.at(1) = -0.1;
        P.at(2) = -offset_eps;
      },
      Access::write(Sym<REAL>("P")))
      ->execute();

  auto groups = cartesian_trajectory_intersection->post_integration(A);

  const int correct_group = 4;
  for (auto &gx : groups) {
    const int correct_npart = gx.first == correct_group ? 1 : 0;
    ASSERT_EQ(gx.second->get_npart_local(), correct_npart);
  }

  cartesian_trajectory_intersection->free();
  sycl_target->free();
  A->domain->mesh->free();
}
