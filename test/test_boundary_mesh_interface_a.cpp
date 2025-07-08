#include "include/test_neso_particles.hpp"

TEST(BoundaryMeshInterface, mpi_neighbours) {

  MPI_Comm comm = MPI_COMM_WORLD;

  BoundaryMeshInterface bmi;
  bmi.boundary_init(comm);

  int rank = -1;
  int size = -1;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  const int num_owned_geoms = 4;

  std::vector<std::pair<int, int>> test_map;
  int num_neighbours = 0;
  if (rank > 0) {
    num_neighbours = rank + 1;
  }

  int correct_num_edges_t = 0;

  std::vector<int> correct_in_degree_t(size);
  std::fill(correct_in_degree_t.begin(), correct_in_degree_t.end(), 0);
  std::vector<int> correct_in_degree(size);
  std::fill(correct_in_degree.begin(), correct_in_degree.end(), 0);

  for (int nx = 0; nx < num_neighbours; nx++) {
    const int rx = nx % size;
    const int gx = rx * num_owned_geoms + (rx + rank) % num_owned_geoms;
    test_map.push_back({rx, gx});
    correct_in_degree_t.at(rx)++;
    correct_num_edges_t++;
  }
  int correct_num_edges = 0;
  MPICHK(MPI_Allreduce(&correct_num_edges_t, &correct_num_edges, 1, MPI_INT,
                       MPI_SUM, comm));

  MPICHK(MPI_Allreduce(correct_in_degree_t.data(), correct_in_degree.data(),
                       size, MPI_INT, MPI_SUM, comm));

  bmi.boundary_extend_exchange_pattern(test_map);

  MPI_Comm ncomm = bmi.boundary.ncomm;
  ASSERT_NE(ncomm, MPI_COMM_NULL);

  int test_size = 0;
  int test_rank = 0;
  MPICHK(MPI_Comm_rank(ncomm, &test_rank));
  MPICHK(MPI_Comm_size(ncomm, &test_size));

  ASSERT_EQ(size, test_size);
  ASSERT_EQ(rank, test_rank);

  int topo_status = 0;
  MPICHK(MPI_Topo_test(ncomm, &topo_status));
  ASSERT_EQ(topo_status, MPI_DIST_GRAPH);

  int indegree = -1;
  int outdegree = -1;
  int weighted = -1;
  MPICHK(
      MPI_Dist_graph_neighbors_count(ncomm, &indegree, &outdegree, &weighted));

  ASSERT_EQ(indegree, correct_in_degree.at(rank));
  ASSERT_EQ(outdegree, num_neighbours);
  ASSERT_EQ(weighted, 0);
}
