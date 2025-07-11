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

  std::map<int, std::vector<std::pair<int, int>>> test_map;
  int num_neighbours = 0;
  if (rank > 0) {
    num_neighbours = rank + 1;
  }

  int correct_num_edges_t = 0;
  std::vector<int> correct_in_degree_t(size);
  std::fill(correct_in_degree_t.begin(), correct_in_degree_t.end(), 0);
  std::vector<int> correct_in_degree(size);
  std::fill(correct_in_degree.begin(), correct_in_degree.end(), 0);

  for (int rankx = 0; rankx < size; rankx++) {
    int num_neighbours_inner = 0;
    if (rankx > 0) {
      num_neighbours_inner = rankx + 1;
    }
    for (int nx = 0; nx < num_neighbours_inner; nx++) {
      const int rx = nx % size;
      const int gx = rx * num_owned_geoms + (rx + rankx) % num_owned_geoms;
      test_map[rankx].push_back({rx, gx});

      if (rankx == rank) {
        correct_in_degree_t.at(rx)++;
        correct_num_edges_t++;
      }
    }
  }

  int correct_num_edges = 0;
  MPICHK(MPI_Allreduce(&correct_num_edges_t, &correct_num_edges, 1, MPI_INT,
                       MPI_SUM, comm));

  MPICHK(MPI_Allreduce(correct_in_degree_t.data(), correct_in_degree.data(),
                       size, MPI_INT, MPI_SUM, comm));

  bmi.boundary_extend_exchange_pattern(test_map[rank]);

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

  std::vector<int> sources(indegree);
  std::vector<int> sourcesweights(indegree);
  std::vector<int> destinations(outdegree);
  std::fill(destinations.begin(), destinations.end(), -1);
  std::vector<int> destweights(outdegree);

  MPICHK(MPI_Dist_graph_neighbors(ncomm, indegree, sources.data(),
                                  sourcesweights.data(), outdegree,
                                  destinations.data(), destweights.data()));

  for (int dx = 0; dx < outdegree; dx++) {
    const int dest_rank = destinations.at(dx);
    ASSERT_TRUE((-1 < dest_rank) && (dest_rank < size));
    ASSERT_TRUE(bmi.boundary.map_recv_rank_to_geom_ids.count(dest_rank));
  }

  for (int sx = 0; sx < indegree; sx++) {
    const int src_rank = sources.at(sx);
    ASSERT_TRUE((-1 < src_rank) && (src_rank < size));
    std::set<int> candidate_ranks;
    for (auto &rank_gids : test_map[src_rank]) {
      candidate_ranks.insert(rank_gids.first);
    }
    ASSERT_TRUE(candidate_ranks.count(rank));
  }

  // Test that the correct number of geoms were communicated
  for (int sx = 0; sx < indegree; sx++) {
    const int src_rank = sources.at(sx);
    const int to_test_count = bmi.boundary.incoming_geom_counts.at(sx);
    int correct_count = 0;
    for (auto &rank_gid : test_map.at(src_rank)) {
      if (rank_gid.first == rank) {
        correct_count++;
      }
    }
    ASSERT_EQ(to_test_count, correct_count);
  }

  std::set<int> incoming_geoms_correct;
  for (int sx = 0; sx < indegree; sx++) {
    const int src_rank = sources.at(sx);
    for (auto &rank_gid : test_map.at(src_rank)) {
      if (rank_gid.first == rank) {
        incoming_geoms_correct.insert(rank_gid.second);
      }
    }
  }
  std::set<int> incoming_geoms_to_test;
  for (int gx : bmi.boundary.incoming_geom_ids) {
    incoming_geoms_to_test.insert(gx);
  }
  ASSERT_EQ(incoming_geoms_to_test, incoming_geoms_correct);

  for (int rx = 0; rx < size; rx++) {
    incoming_geoms_correct.clear();
    for (auto &rank_gid : test_map[rx]) {
      if (rank_gid.first == rank) {
        incoming_geoms_correct.insert(rank_gid.second);
      }
    }
    ASSERT_EQ(incoming_geoms_correct,
              bmi.boundary.map_send_rank_to_geom_ids[rx]);
  }
}
