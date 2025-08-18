#include "include/test_neso_particles.hpp"

TEST(BoundaryTruncation, 2d) {
  const int ncell_x = 12;
  const int ncell_y = 7;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_2d(4, ncell_x, ncell_y);

  auto sycl_target = sycl_target_t;
  auto A = A_t;

  A->add_particle_dat(Sym<REAL>("TSP"), 2);
  A->add_particle_dat(Sym<REAL>("PR"), 2);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0, 1, 2, 3};

  auto cartesian_trajectory_intersection =
      std::make_shared<CartesianTrajectoryIntersection>(
          sycl_target,
          std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
          boundary_groups);
  cartesian_trajectory_intersection->prepare_particle_group(A);

  constexpr REAL dt = 100.0;
  cartesian_trajectory_intersection->pre_integration(A);
  particle_loop(
      A,
      [=](auto P, auto V, auto PR) {
        const REAL magnitude2 = V.at(0) * V.at(0) + V.at(1) * V.at(1);
        const bool is_small = magnitude2 < 1.0e-12;
        const REAL scaling = is_small ? 1.0 : Kernel::rsqrt(magnitude2);
        V.at(0) = is_small ? 0.7071067811865475 : V.at(0) * scaling;
        V.at(1) = is_small ? 0.7071067811865475 : V.at(1) * scaling;
        P.at(0) += dt * V.at(0);
        P.at(1) += dt * V.at(1);
        PR.at(0) = P.at(0);
        PR.at(1) = P.at(1);
      },
      Access::write(Sym<REAL>("P")), Access::write(Sym<REAL>("V")),
      Access::write(Sym<REAL>("PR")))
      ->execute();

  particle_loop(
      A,
      [=](auto TSP) {
        TSP.at(0) = 3.0 * dt;
        TSP.at(1) = dt;
      },
      Access::write(Sym<REAL>("TSP")))
      ->execute();

  auto groups = cartesian_trajectory_intersection->post_integration(A);

  ASSERT_EQ(groups.at(0)->get_npart_local(), A->get_npart_local());

  constexpr REAL reset_distance = std::numeric_limits<REAL>::epsilon();
  auto truncation = std::make_shared<BoundaryTruncation>(2, reset_distance);

  truncation->execute(groups.at(0), Sym<REAL>("P"), Sym<REAL>("TSP"),
                      cartesian_trajectory_intersection->previous_position_sym);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      groups.at(0),
      [=](auto P, auto TSP, auto PR, auto PP) {
        NESO_KERNEL_ASSERT((0.0 <= P.at(0)) && (P.at(0) <= ncell_x), k_ep);
        NESO_KERNEL_ASSERT((0.0 <= P.at(1)) && (P.at(1) <= ncell_y), k_ep);

        const REAL f0 = PR.at(0) - PP.at(0);
        const REAL f1 = PR.at(1) - PP.at(1);
        const REAL t0 = P.at(0) - PP.at(0);
        const REAL t1 = P.at(1) - PP.at(1);

        const REAL fnorm = Kernel::sqrt(f0 * f0 + f1 * f1);
        NESO_KERNEL_ASSERT(Kernel::abs(fnorm - dt) < 1.0e-10, k_ep);

        const REAL tnorm = Kernel::sqrt(t0 * t0 + t1 * t1);
        const REAL correct_tsp = dt * tnorm / fnorm;

        NESO_KERNEL_ASSERT(
            Kernel::abs(TSP.at(0) - (2.0 * dt + correct_tsp)) < 1.0e-12, k_ep);
        NESO_KERNEL_ASSERT(Kernel::abs(TSP.at(1) - correct_tsp) < 1.0e-12,
                           k_ep);
      },
      Access::read(Sym<REAL>("P")), Access::read(Sym<REAL>("TSP")),
      Access::read(Sym<REAL>("PR")),
      Access::read(cartesian_trajectory_intersection->previous_position_sym))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  particle_loop(
      groups.at(0),
      [=](auto P) {
        constexpr REAL tol = 2.0 * reset_distance;
        const bool near_xl = Kernel::abs(P.at(0)) <= tol;
        const bool near_xu = Kernel::abs(P.at(0) - ncell_x) <= tol;
        const bool near_yl = Kernel::abs(P.at(1)) <= tol;
        const bool near_yu = Kernel::abs(P.at(1) - ncell_y) <= tol;
        const bool near_found = near_xl || near_xu || near_yl || near_yu;
        NESO_KERNEL_ASSERT(near_found, k_ep);
      },
      Access::read(Sym<REAL>("P")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
  A_t->domain->mesh->free();
}

TEST(BoundaryTruncation, 3d) {
  const int ncell_x = 12;
  const int ncell_y = 7;
  const int ncell_z = 5;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_3d(4, ncell_x, ncell_y, ncell_z);

  auto sycl_target = sycl_target_t;
  auto A = A_t;

  A->add_particle_dat(Sym<REAL>("TSP"), 2);
  A->add_particle_dat(Sym<REAL>("PR"), 3);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0, 1, 2, 3, 4, 5};

  auto cartesian_trajectory_intersection =
      std::make_shared<CartesianTrajectoryIntersection>(
          sycl_target,
          std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh),
          boundary_groups);
  cartesian_trajectory_intersection->prepare_particle_group(A);

  constexpr REAL dt = 100.0;
  cartesian_trajectory_intersection->pre_integration(A);
  particle_loop(
      A,
      [=](auto P, auto V, auto PR) {
        const REAL magnitude2 =
            V.at(0) * V.at(0) + V.at(1) * V.at(1) + V.at(2) * V.at(2);
        const bool is_small = magnitude2 < 1.0e-12;
        const REAL scaling = is_small ? 1.0 : Kernel::rsqrt(magnitude2);
        V.at(0) = is_small ? 0.5773502691896258 : V.at(0) * scaling;
        V.at(1) = is_small ? 0.5773502691896258 : V.at(1) * scaling;
        V.at(2) = is_small ? 0.5773502691896258 : V.at(2) * scaling;
        P.at(0) += dt * V.at(0);
        P.at(1) += dt * V.at(1);
        P.at(2) += dt * V.at(2);
        PR.at(0) = P.at(0);
        PR.at(1) = P.at(1);
        PR.at(2) = P.at(2);
      },
      Access::write(Sym<REAL>("P")), Access::write(Sym<REAL>("V")),
      Access::write(Sym<REAL>("PR")))
      ->execute();

  particle_loop(
      A,
      [=](auto TSP) {
        TSP.at(0) = 3.0 * dt;
        TSP.at(1) = dt;
      },
      Access::write(Sym<REAL>("TSP")))
      ->execute();

  auto groups = cartesian_trajectory_intersection->post_integration(A);

  ASSERT_EQ(groups.at(0)->get_npart_local(), A->get_npart_local());

  constexpr REAL reset_distance = std::numeric_limits<REAL>::epsilon();
  auto truncation = std::make_shared<BoundaryTruncation>(3, reset_distance);
  truncation->execute(groups.at(0), Sym<REAL>("P"), Sym<REAL>("TSP"),
                      cartesian_trajectory_intersection->previous_position_sym);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      groups.at(0),
      [=](auto P, auto TSP, auto PR, auto PP) {
        NESO_KERNEL_ASSERT((0.0 <= P.at(0)) && (P.at(0) <= ncell_x), k_ep);
        NESO_KERNEL_ASSERT((0.0 <= P.at(1)) && (P.at(1) <= ncell_y), k_ep);
        NESO_KERNEL_ASSERT((0.0 <= P.at(2)) && (P.at(2) <= ncell_z), k_ep);

        const REAL f0 = PR.at(0) - PP.at(0);
        const REAL f1 = PR.at(1) - PP.at(1);
        const REAL f2 = PR.at(2) - PP.at(2);
        const REAL t0 = P.at(0) - PP.at(0);
        const REAL t1 = P.at(1) - PP.at(1);
        const REAL t2 = P.at(2) - PP.at(2);

        const REAL fnorm = Kernel::sqrt(f0 * f0 + f1 * f1 + f2 * f2);
        NESO_KERNEL_ASSERT(Kernel::abs(fnorm - dt) < 1.0e-10, k_ep);

        const REAL tnorm = Kernel::sqrt(t0 * t0 + t1 * t1 + t2 * t2);
        const REAL correct_tsp = dt * tnorm / fnorm;

        NESO_KERNEL_ASSERT(
            Kernel::abs(TSP.at(0) - (2.0 * dt + correct_tsp)) < 1.0e-12, k_ep);
        NESO_KERNEL_ASSERT(Kernel::abs(TSP.at(1) - correct_tsp) < 1.0e-12,
                           k_ep);
      },
      Access::read(Sym<REAL>("P")), Access::read(Sym<REAL>("TSP")),
      Access::read(Sym<REAL>("PR")),
      Access::read(cartesian_trajectory_intersection->previous_position_sym))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  particle_loop(
      groups.at(0),
      [=](auto P) {
        constexpr REAL tol = 2.0 * reset_distance;
        const bool near_xl = Kernel::abs(P.at(0)) <= tol;
        const bool near_xu = Kernel::abs(P.at(0) - ncell_x) <= tol;
        const bool near_yl = Kernel::abs(P.at(1)) <= tol;
        const bool near_yu = Kernel::abs(P.at(1) - ncell_y) <= tol;
        const bool near_zl = Kernel::abs(P.at(2)) <= tol;
        const bool near_zu = Kernel::abs(P.at(2) - ncell_z) <= tol;
        const bool near_found =
            near_xl || near_xu || near_yl || near_yu || near_zl || near_zu;
        NESO_KERNEL_ASSERT(near_found, k_ep);
      },
      Access::read(Sym<REAL>("P")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
  A_t->domain->mesh->free();
}
