#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST(MeshHierarchy, test_mesh_hierarchy_1) {

  SYCLTarget sycl_target{GPU_SELECTOR, MPI_COMM_WORLD};

  std::vector<int> dims(2);
  dims[0] = 2;
  dims[1] = 4;

  MeshHierarchy mh(sycl_target, 2, dims, 2.0, 2);

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

  SYCLTarget sycl_target{GPU_SELECTOR, MPI_COMM_WORLD};

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  CartesianHMesh mesh(sycl_target, ndim, dims, cell_extent, subdivision_order);
  MeshHierarchy mh(mesh);

    






}
