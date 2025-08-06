#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

TEST(Communication, edges_counter) {
  CommunicationEdgesCounter counter(MPI_COMM_WORLD);

  int rank, size;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  {
    std::vector<int> remotes = {(rank + 1) % size, (rank + 3) % size};
    counter.reset();
    counter.init_count(remotes);
    const int num = counter.get_count();
    ASSERT_EQ(num, 2);
  }

  {
    std::vector<int> remotes = {
        0,
    };
    counter.reset();
    counter.init_count(remotes);
    const int num = counter.get_count();
    if (rank == 0) {
      ASSERT_EQ(num, size);
    } else {
      ASSERT_EQ(num, 0);
    }
  }

  counter.free();
}

TEST(Communication, edges_ranks) {
  CommunicationEdgesCounter counter(MPI_COMM_WORLD);

  int rank, size;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  {
    std::vector<int> recv_ranks = {(rank + 1) % size, (rank + 3) % size};
    std::vector<std::int64_t> recv_data = {(rank + 1), (rank + 1)};

    std::vector<int> send_ranks;
    std::vector<std::int64_t> send_data;

    counter.get_remote_ranks(recv_ranks, recv_data, send_ranks, send_data);

    std::vector<int> correct_send_ranks = {
        (rank - 1 + 10 * size) % size,
        (rank - 3 + 10 * size) % size,
    };

    for (auto rx : correct_send_ranks) {
      auto idx = std::find(send_ranks.begin(), send_ranks.end(), rx);
      ASSERT_TRUE(idx != send_ranks.end());
      const int id = idx - send_ranks.begin();
      ASSERT_EQ(send_data.at(id), rx + 1);
    }
  }

  {
    std::vector<int> recv_ranks = {
        0,
    };
    std::vector<std::int64_t> recv_data = {
        (rank + 1),
    };

    std::vector<int> send_ranks;
    std::vector<std::int64_t> send_data;
    counter.get_remote_ranks(recv_ranks, recv_data, send_ranks, send_data);

    if (rank == 0) {

      std::vector<int> correct_send_ranks(size);
      std::iota(correct_send_ranks.begin(), correct_send_ranks.end(), 0);

      for (auto rx : correct_send_ranks) {
        auto idx = std::find(send_ranks.begin(), send_ranks.end(), rx);
        ASSERT_TRUE(idx != send_ranks.end());
        const int id = idx - send_ranks.begin();
        ASSERT_EQ(send_data.at(id), rx + 1);
      }

    } else {
      ASSERT_EQ(send_ranks.size(), 0);
      ASSERT_EQ(send_data.size(), 0);
    }
  }

  counter.free();
}

TEST(communication_utility, gather_v) {

  int rank, size;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  const int offset = 3;
  const int N = rank + offset;
  std::vector<REAL> input(N);
  std::iota(input.begin(), input.end(), N * rank);

  std::vector<REAL> output;

  const int root = size - 1;
  const bool root_rank = rank == root;
  gather_v(input, MPI_COMM_WORLD, root, output);

  if (root_rank) {
    int total_size = 0;
    for (int rx = 0; rx < size; rx++) {
      total_size += rx + offset;
    }

    ASSERT_EQ(output.size(), total_size);
    int index = 0;
    for (int rx = 0; rx < size; rx++) {
      const int start_rank = (rx + offset) * rx;
      for (int ix = 0; ix < (rx + offset); ix++) {
        ASSERT_EQ(output.at(index), start_rank + ix);
        index++;
      }
    }
  }
}

TEST(communication_utility, all_gather_v) {

  int rank, size;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  const int offset = 5;
  const int N = rank + offset;
  std::vector<REAL> input(N);
  std::iota(input.begin(), input.end(), N * rank);

  std::vector<REAL> output;

  all_gather_v(input, MPI_COMM_WORLD, output);

  int total_size = 0;
  for (int rx = 0; rx < size; rx++) {
    total_size += rx + offset;
  }

  ASSERT_EQ(output.size(), total_size);
  int index = 0;
  for (int rx = 0; rx < size; rx++) {
    const int start_rank = (rx + offset) * rx;
    for (int ix = 0; ix < (rx + offset); ix++) {
      ASSERT_EQ(output.at(index), start_rank + ix);
      index++;
    }
  }
}

TEST(communication_utility, reverse_graph_edge_directions) {

  int rank, size;
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  MPI_Comm ncomm;

  int degrees = 1;
  int destinations = 0;

  MPICHK(MPI_Dist_graph_create(MPI_COMM_WORLD, 1, &rank, &degrees,
                               &destinations, MPI_UNWEIGHTED, MPI_INFO_NULL, 0,
                               &ncomm));

  {
    int indegree = -1, outdegree = -1, weighted = -1;

    MPICHK(MPI_Dist_graph_neighbors_count(ncomm, &indegree, &outdegree,
                                          &weighted));

    ASSERT_EQ(outdegree, 1);
    ASSERT_EQ(indegree, rank == 0 ? size : 0);

    std::vector<int> sources(indegree);
    std::vector<int> sourcesweights(indegree);
    std::vector<int> destinations(outdegree);
    std::vector<int> destweights(outdegree);

    MPICHK(MPI_Dist_graph_neighbors(ncomm, indegree, sources.data(),
                                    sourcesweights.data(), outdegree,
                                    destinations.data(), destweights.data()));

    if (rank == 0) {
      std::set<int> set_sources;
      for (int ix : sources) {
        set_sources.insert(ix);
      }
      ASSERT_EQ(set_sources.size(), size);
    } else {
      ASSERT_EQ(destinations.at(0), 0);
    }
  }

  MPI_Comm rncomm;
  ASSERT_EQ(reverse_graph_edge_directions(MPI_COMM_WORLD, ncomm, &rncomm),
            MPI_SUCCESS);

  {
    int indegree = -1, outdegree = -1, weighted = -1;

    MPICHK(MPI_Dist_graph_neighbors_count(rncomm, &indegree, &outdegree,
                                          &weighted));

    ASSERT_EQ(outdegree, rank == 0 ? size : 0);
    ASSERT_EQ(indegree, 1);

    std::vector<int> sources(indegree);
    std::vector<int> sourcesweights(indegree);
    std::vector<int> destinations(outdegree);
    std::vector<int> destweights(outdegree);

    MPICHK(MPI_Dist_graph_neighbors(rncomm, indegree, sources.data(),
                                    sourcesweights.data(), outdegree,
                                    destinations.data(), destweights.data()));

    if (rank == 0) {
      std::set<int> set_destinations;
      for (int ix : destinations) {
        set_destinations.insert(ix);
      }
      ASSERT_EQ(set_destinations.size(), size);
    } else {
      ASSERT_EQ(sources.at(0), 0);
    }
  }
}
