#include "include/test_neso_particles.hpp"

TEST(BoundaryTruncation, 2d) {
  const int ncell_x = 12;
  const int ncell_y = 7;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_2d(4, ncell_x, ncell_y);

  auto sycl_target = sycl_target_t;
  auto A = A_t;

  const int ndim = 2;
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
  
  particle_loop(
    A,
    [=](auto V){


    },
    Access::write(Sym<REAL>("V"))
  )->execute();





  sycl_target->free();
  A_t->domain->mesh->free();
}
