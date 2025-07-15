#include "include/test_neso_particles.hpp"

TEST(CartesianHMesh, mpi_topology_2d) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 7;
  dims[1] = 19;

  const double cell_extent = 1.0;
  const int subdivision_order = 1;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  // Test the mapping from cells to owning ranks
  auto comm = mesh->get_comm();
  int rank = 0;
  MPICHK(MPI_Comm_rank(comm, &rank));
  auto cell_starts = mesh->cell_starts;
  auto cell_ends = mesh->cell_starts;
  for (int cy = cell_starts[1]; cy < cell_ends[1]; cy++) {
    for (int cx = cell_starts[0]; cx < cell_ends[0]; cx++) {
      INT mesh_index[2] = {cx, cy};
      int rank_to_test = mesh->get_mesh_tuple_owning_rank(mesh_index);
      ASSERT_EQ(rank, rank_to_test);
    }
  }

  // Test the mapping from cell faces to owning cells
  auto lambda_test = [&](std::array<INT, 2> face, std::array<INT, 2> cell) {
    INT face_tuple[2] = {face[0], face[1]};
    INT cell_tuple[2] = {-1, -1};
    mesh->get_mesh_tuple_owning_face_tuple(face_tuple, cell_tuple);

    for (int dx = 0; dx < ndim; dx++) {
      ASSERT_EQ(cell_tuple[dx], cell[dx]);
    }

    const auto linear_index =
        mesh->get_face_linear_index_from_tuple(face.data());

    INT face_to_test[3] = {0, 0, 0};

    mesh->get_face_id_as_tuple(linear_index, face_to_test);

    for (int dx = 0; dx < ndim; dx++) {
      ASSERT_EQ(face[dx], face_to_test[dx]);
    }

    ASSERT_EQ(mesh->get_face_id_owning_rank(linear_index),
              mesh->get_mesh_tuple_owning_rank(cell.data()));
  };

  lambda_test({0, 0}, {0, 0});
  lambda_test({0, 1}, {1, 0});

  lambda_test({1, 0}, {mesh->cell_counts[0] - 1, 0});
  lambda_test({1, 1}, {mesh->cell_counts[0] - 1, 1});
  lambda_test({1, 5}, {mesh->cell_counts[0] - 1, 5});

  lambda_test({2, 0}, {0, mesh->cell_counts[1] - 1});
  lambda_test({2, 1}, {1, mesh->cell_counts[1] - 1});
  lambda_test({2, 3}, {3, mesh->cell_counts[1] - 1});
  lambda_test({2, 13}, {13, mesh->cell_counts[1] - 1});

  lambda_test({3, 0}, {0, 0});
  lambda_test({3, 1}, {0, 1});
  lambda_test({3, 5}, {0, 5});

  // test the linear to tuple methods
  const int cell_count = mesh->get_cell_count();
  auto cells = mesh->get_owned_cells();

  for (int cx = 0; cx < cell_count; cx++) {
    auto correct = cells.at(cx);
    auto to_test = mesh->get_global_cell_tuple_index(cx);

    ASSERT_EQ(correct, to_test);
  }

  mesh->free();
}

TEST(CartesianHMesh, mpi_topology_3d) {

  const int ndim = 3;
  std::vector<int> dims(ndim);
  dims[0] = 7;
  dims[1] = 19;
  dims[2] = 5;

  const double cell_extent = 1.0;
  const int subdivision_order = 0;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto comm = mesh->get_comm();
  int rank = 0;
  MPICHK(MPI_Comm_rank(comm, &rank));

  auto cell_starts = mesh->cell_starts;
  auto cell_ends = mesh->cell_starts;
  for (int cz = cell_starts[2]; cz < cell_ends[2]; cz++) {
    for (int cy = cell_starts[1]; cy < cell_ends[1]; cy++) {
      for (int cx = cell_starts[0]; cx < cell_ends[0]; cx++) {
        INT mesh_index[3] = {cx, cy, cz};
        int rank_to_test = mesh->get_mesh_tuple_owning_rank(mesh_index);
        ASSERT_EQ(rank, rank_to_test);
      }
    }
  }

  // Test the mapping from cell faces to owning cells
  auto lambda_test = [&](std::array<INT, 3> face, std::array<INT, 3> cell) {
    INT face_tuple[3] = {face[0], face[1], face[2]};
    INT cell_tuple[3] = {-1, -1, -1};
    mesh->get_mesh_tuple_owning_face_tuple(face_tuple, cell_tuple);

    for (int dx = 0; dx < ndim; dx++) {
      ASSERT_EQ(cell_tuple[dx], cell[dx]);
    }

    const auto linear_index =
        mesh->get_face_linear_index_from_tuple(face.data());
    INT face_to_test[3] = {0, 0, 0};

    mesh->get_face_id_as_tuple(linear_index, face_to_test);
    for (int dx = 0; dx < ndim; dx++) {
      ASSERT_EQ(face[dx], face_to_test[dx]);
    }

    ASSERT_EQ(mesh->get_face_id_owning_rank(linear_index),
              mesh->get_mesh_tuple_owning_rank(cell.data()));
  };

  lambda_test({0, 0, 0}, {0, 0, 0});
  lambda_test({0, 3, 0}, {3, 0, 0});
  lambda_test({0, 5, 3}, {5, 0, 3});
  lambda_test({0, 0, 2}, {0, 0, 2});

  lambda_test({1, 0, 0}, {mesh->cell_counts[0] - 1, 0, 0});
  lambda_test({1, 3, 0}, {mesh->cell_counts[0] - 1, 3, 0});
  lambda_test({1, 0, 2}, {mesh->cell_counts[0] - 1, 0, 2});
  lambda_test({1, 4, 3}, {mesh->cell_counts[0] - 1, 4, 3});

  lambda_test({2, 0, 0}, {0, mesh->cell_counts[1] - 1, 0});
  lambda_test({2, 3, 0}, {3, mesh->cell_counts[1] - 1, 0});
  lambda_test({2, 0, 2}, {0, mesh->cell_counts[1] - 1, 2});
  lambda_test({2, 5, 3}, {5, mesh->cell_counts[1] - 1, 3});

  lambda_test({3, 0, 0}, {0, 0, 0});
  lambda_test({3, 2, 0}, {0, 2, 0});
  lambda_test({3, 0, 3}, {0, 0, 3});
  lambda_test({3, 5, 3}, {0, 5, 3});

  lambda_test({4, 0, 0}, {0, 0, 0});
  lambda_test({4, 3, 0}, {3, 0, 0});
  lambda_test({4, 0, 5}, {0, 5, 0});
  lambda_test({4, 6, 4}, {6, 4, 0});

  lambda_test({5, 0, 0}, {0, 0, mesh->cell_counts[2] - 1});
  lambda_test({5, 3, 0}, {3, 0, mesh->cell_counts[2] - 1});
  lambda_test({5, 0, 5}, {0, 5, mesh->cell_counts[2] - 1});
  lambda_test({5, 6, 4}, {6, 4, mesh->cell_counts[2] - 1});

  // test the linear to tuple methods
  const int cell_count = mesh->get_cell_count();
  auto cells = mesh->get_owned_cells();

  for (int cx = 0; cx < cell_count; cx++) {
    auto correct = cells.at(cx);
    auto to_test = mesh->get_global_cell_tuple_index(cx);

    ASSERT_EQ(correct, to_test);
  }

  mesh->free();
}
