#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST(MeshHierarchy, test_mesh_hierarchy_creation) {

  SYCLTarget sycl_target{GPU_SELECTOR, MPI_COMM_WORLD};

  std::vector<int> dims(2);
  dims[0] = 2;
  dims[1] = 4;
  std::vector<double> origin(2);
  origin[0] = 0.0;
  origin[1] = 0.0;

  MeshHierarchy mh(MPI_COMM_WORLD, 2, dims, origin, 2.0, 2);

  ASSERT_TRUE(mh.ndim == 2);
  ASSERT_TRUE(mh.dims[0] == 2);
  ASSERT_TRUE(mh.dims[1] == 4);
  ASSERT_TRUE(mh.subdivision_order == 2);
  ASSERT_TRUE(mh.cell_width_coarse == 2.0);
  ASSERT_TRUE(std::abs(mh.cell_width_fine - 2.0 / std::pow(2, 2)) < 1E-14);
  ASSERT_TRUE(mh.ncells_coarse == 8);
  ASSERT_TRUE(mh.ncells_fine == 16);

  mh.free();
  sycl_target.free();
}

TEST(MeshHierarchy, test_mesh_hierarchy_indexing) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order);

  // test conversion from tuple indices to linear indices.
  INT index[4] = {0, 0, 0, 0};

  auto mh = mesh.get_mesh_hierarchy();
  ASSERT_TRUE(mh->tuple_to_linear_global(index) == 0);
  ASSERT_TRUE(mh->tuple_to_linear_coarse(index) == 0);
  ASSERT_TRUE(mh->tuple_to_linear_fine(index) == 0);

  index[0] = 3;
  index[1] = 7;
  index[2] = 2;
  index[3] = 3;

  ASSERT_TRUE(mh->tuple_to_linear_coarse(index) == 31);
  ASSERT_TRUE(mh->tuple_to_linear_fine(index + 2) == 14);
  ASSERT_TRUE(mh->tuple_to_linear_global(index) == 31 * 16 + 14);

  index[0] = 0;
  index[1] = 0;
  index[2] = 0;
  index[3] = 0;
  mh->linear_to_tuple_fine(14, index + 2);
  ASSERT_EQ(index[2], 2);
  ASSERT_EQ(index[3], 3);
  mh->linear_to_tuple_coarse(31, index);
  ASSERT_EQ(index[0], 3);
  ASSERT_EQ(index[1], 7);

  index[0] = 0;
  index[1] = 0;
  index[2] = 0;
  index[3] = 0;
  mh->linear_to_tuple_global(31 * 16 + 14, index);
  ASSERT_EQ(index[0], 3);
  ASSERT_EQ(index[1], 7);
  ASSERT_EQ(index[2], 2);
  ASSERT_EQ(index[3], 3);

  mesh.free();
}

TEST(MeshHierarchy, test_mesh_hierarchy_ownership) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order);

  auto mh = mesh.get_mesh_hierarchy();

  int rank;
  MPICHK(MPI_Comm_rank(mesh.get_comm(), &rank));

  // with this mesh type the ownership is unambiguous.
  // mesh tuple index
  INT index_mesh[3];
  // mesh_hierarchy tuple index
  INT index_mh[6];
  // loop over owned cells
  for (int cz = mesh.cell_starts[2]; cz < mesh.cell_ends[2]; cz++) {
    index_mesh[2] = cz;
    for (int cy = mesh.cell_starts[1]; cy < mesh.cell_ends[1]; cy++) {
      index_mesh[1] = cy;
      for (int cx = mesh.cell_starts[0]; cx < mesh.cell_ends[0]; cx++) {
        index_mesh[0] = cx;
        // convert mesh tuple index to mesh hierarchy tuple index
        mesh.mesh_tuple_to_mh_tuple(index_mesh, index_mh);
        // convert mesh hierarchy tuple index to global linear index in the
        // MeshHierarchy
        const INT index_global = mh->tuple_to_linear_global(index_mh);

        const int claimed_rank = mh->get_owner(index_global);
        ASSERT_EQ(rank, claimed_rank);
      }
    }
  }

  mesh.free();
}
