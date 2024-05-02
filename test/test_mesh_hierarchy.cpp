#include <gtest/gtest.h>
#include <neso_particles.hpp>
using namespace NESO::Particles;

TEST(MeshHierarchy, test_mesh_hierarchy_creation) {

  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);

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
  sycl_target->free();
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

TEST(MeshHierarchyMappers, cart_no_trunc) {

  const int ndim = 3;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;
  dims[2] = 6;
  std::vector<double> origin(ndim);
  origin[0] = 0.5;
  origin[1] = 0;
  origin[2] = 0;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;

  auto mh = std::make_shared<MeshHierarchy>(MPI_COMM_WORLD, ndim, dims, origin,
                                            cell_extent, subdivision_order);

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  auto mapper = std::make_shared<MeshHierarchyMapper>(sycl_target, mh);
  auto host_mapper = mapper->get_host_mapper();

  REAL position[3];
  INT cell[3];

  position[0] = 1.1;
  position[1] = 1.1;
  position[2] = 1.6;
  host_mapper.map_to_cart_tuple_no_trunc(position, cell);
  ASSERT_EQ(cell[0], 1);
  ASSERT_EQ(cell[1], 2);
  ASSERT_EQ(cell[2], 3);

  INT mh_cell0[6];
  INT mh_cell1[6];
  host_mapper.map_to_tuple(position, mh_cell0);
  host_mapper.cart_tuple_to_tuple(cell, mh_cell1);

  ASSERT_EQ(mh_cell0[0], mh_cell1[0]);
  ASSERT_EQ(mh_cell0[1], mh_cell1[1]);
  ASSERT_EQ(mh_cell0[2], mh_cell1[2]);
  ASSERT_EQ(mh_cell0[3], mh_cell1[3]);
  ASSERT_EQ(mh_cell0[4], mh_cell1[4]);
  ASSERT_EQ(mh_cell0[5], mh_cell1[5]);

  position[0] = 5.1;
  position[1] = 9.1;
  position[2] = 7.6;
  host_mapper.map_to_cart_tuple_no_trunc(position, cell);
  ASSERT_EQ(cell[0], 4 * 2 + 1);
  ASSERT_EQ(cell[1], 8 * 2 + 2);
  ASSERT_EQ(cell[2], 6 * 2 + 3);

  position[0] = 0.1;
  position[1] = -1.1;
  position[2] = 7.6;
  host_mapper.map_to_cart_tuple_no_trunc(position, cell);
  ASSERT_EQ(cell[0], -1);
  ASSERT_EQ(cell[1], -3);
  ASSERT_EQ(cell[2], 6 * 2 + 3);

  mh->free();
}

TEST(MeshHierarchyMappers, indexing) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  CartesianHMesh mesh(MPI_COMM_WORLD, ndim, dims, cell_extent,
                      subdivision_order);

  auto mh = mesh.get_mesh_hierarchy();

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  auto mapper = std::make_shared<MeshHierarchyMapper>(sycl_target, mh);
  auto host_mapper = mapper->get_host_mapper();

  INT index[4] = {0, 0, 0, 0};
  ASSERT_EQ(host_mapper.tuple_to_linear_global(index), 0);

  index[0] = 3;
  index[1] = 7;
  index[2] = 2;
  index[3] = 3;

  ASSERT_EQ(mh->tuple_to_linear_coarse(index),
            host_mapper.tuple_to_linear_coarse(index));
  ASSERT_EQ(mh->tuple_to_linear_fine(index + 2),
            host_mapper.tuple_to_linear_fine(index + 2));
  ASSERT_EQ(mh->tuple_to_linear_global(index),
            host_mapper.tuple_to_linear_global(index));

  INT cart_index[2] = {0, 0};
  host_mapper.cart_tuple_to_tuple(cart_index, index);
  ASSERT_EQ(index[0], 0);
  ASSERT_EQ(index[1], 0);
  ASSERT_EQ(index[2], 0);
  ASSERT_EQ(index[3], 0);

  cart_index[0] = 5;
  cart_index[1] = 10;
  host_mapper.cart_tuple_to_tuple(cart_index, index);
  ASSERT_EQ(index[0], 1);
  ASSERT_EQ(index[1], 2);
  ASSERT_EQ(index[2], 1);
  ASSERT_EQ(index[3], 2);

  mesh.free();
}
