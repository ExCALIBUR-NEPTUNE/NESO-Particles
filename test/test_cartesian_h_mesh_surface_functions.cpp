#include "include/test_neso_particles.hpp"

TEST(CartesianHMesh, surface_functions_2d) {

  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  auto mesh = std::dynamic_pointer_cast<CartesianHMesh>(A->domain->mesh);

  std::map<int, std::vector<int>> boundary_groups;
  boundary_groups[0] = {0, 1, 2};
  boundary_groups[1] = {3};

  CartesianTrajectoryIntersection cti(sycl_target, mesh, boundary_groups,
                                      1.0e-14);

  auto u0 = cti.create_function(0, "DG", 0);

  const std::size_t num_geoms = mesh->get_all_face_cells_on_face(0).size() +
                                mesh->get_all_face_cells_on_face(1).size() +
                                mesh->get_all_face_cells_on_face(2).size();

  ASSERT_EQ(u0->d_dofs->size, num_geoms);
  ASSERT_EQ(u0->local_dof_count, num_geoms);
  ASSERT_EQ(u0->cell_dof_count, 1);
  ASSERT_EQ(u0->mesh, mesh);
  ASSERT_EQ(u0->sycl_target, sycl_target);
  ASSERT_EQ(u0->ndim, 1);
  ASSERT_EQ(u0->cell_count, num_geoms);
  ASSERT_EQ(u0->function_space, "DG");
  ASSERT_EQ(u0->polynomial_order, 0);
  ASSERT_EQ(u0->cells.size(), num_geoms);
  ASSERT_EQ(u0->element_group, 0);

  cti.free();
  mesh->free();
}
