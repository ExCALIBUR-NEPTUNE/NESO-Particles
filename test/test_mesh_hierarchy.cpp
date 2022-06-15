#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST(MeshHierarchy, test_mesh_hierarchy_1) {

  SYCLTarget sycl_target{GPU_SELECTOR, MPI_COMM_WORLD};

  std::vector<int> dims(2);
  dims[0] = 2;
  dims[1] = 4;

  MeshHierarchy mh(MPI_COMM_WORLD, 2, dims, 2.0, 2);

  ASSERT_TRUE(mh.ndim == 2);
  ASSERT_TRUE(mh.dims[0] == 2);
  ASSERT_TRUE(mh.dims[1] == 4);
  ASSERT_TRUE(mh.subdivision_order == 2);
  ASSERT_TRUE(mh.cell_width_coarse == 2.0);
  ASSERT_TRUE(std::abs(mh.cell_width_fine - 2.0 / std::pow(2, 2)) < 1E-14);
  ASSERT_TRUE(mh.ncells_coarse == 8);
  ASSERT_TRUE(mh.ncells_fine == 16);
}

TEST(MeshHierarchy, test_mesh_hierarchy_2) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order);

  // test conversion from tuple indices to linear indices.
  int index[4] = {0, 0, 0, 0};

  auto mh = mesh.get_mesh_hierarchy();
  ASSERT_TRUE(mh.tuple_to_linear_global(index) == 0);
  ASSERT_TRUE(mh.tuple_to_linear_coarse(index) == 0);
  ASSERT_TRUE(mh.tuple_to_linear_fine(index) == 0);

  index[0] = 3;
  index[1] = 7;
  index[2] = 2;
  index[3] = 3;

  ASSERT_TRUE(mh.tuple_to_linear_coarse(index) == 31);
  ASSERT_TRUE(mh.tuple_to_linear_fine(index + 2) == 14);
  ASSERT_TRUE(mh.tuple_to_linear_global(index) == 31 * 16 + 14);

  index[0] = 0;
  index[1] = 0;
  index[2] = 0;
  index[3] = 0;
  mh.linear_to_tuple_fine(14, index + 2);
  ASSERT_EQ(index[2], 2);
  ASSERT_EQ(index[3], 3);
  mh.linear_to_tuple_coarse(31, index);
  ASSERT_EQ(index[0], 3);
  ASSERT_EQ(index[1], 7);

  index[0] = 0;
  index[1] = 0;
  index[2] = 0;
  index[3] = 0;
  mh.linear_to_tuple_global(31 * 16 + 14, index);
  ASSERT_EQ(index[0], 3);
  ASSERT_EQ(index[1], 7);
  ASSERT_EQ(index[2], 2);
  ASSERT_EQ(index[3], 3);
}
