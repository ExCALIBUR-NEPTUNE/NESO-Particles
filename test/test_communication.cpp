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

TEST(communication_utility, set_communication_pairwise) {
  int rank = 0;
  int size = 0;

  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (size % 2 == 0) {

    std::set<int> correct;
    for (int ix = 0; ix < rank; ix++) {
      correct.insert((ix + 3) % 7);
    }

    const bool is_sending_rank = rank % 2 == 0;

    if (is_sending_rank) {
      const int recv_rank = rank + 1;

      set_send(correct, recv_rank, 4, MPI_COMM_WORLD);
      auto to_test = set_recv<int>(recv_rank, 5, MPI_COMM_WORLD);
      ASSERT_EQ(correct, to_test);
    } else {
      const int send_rank = rank - 1;

      auto set_incomming = set_recv<int>(send_rank, 4, MPI_COMM_WORLD);
      set_send(set_incomming, send_rank, 5, MPI_COMM_WORLD);
    }
  }
}

TEST(communication_utility, set_communication_bcast) {
  int rank = 0;
  int size = 0;

  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  std::set<int> correct;

  for (int ix = 0; ix < 100; ix++) {
    correct.insert((ix * 7 + 9) % 13);
  }

  auto to_test = set_bcast(correct, size - 1, MPI_COMM_WORLD);

  ASSERT_EQ(correct, to_test);
}

TEST(communication_utility, set_reduce_union) {
  int rank = 0;
  int size = 0;

  MPICHK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  auto lambda_get_contrib = [](int rank) {
    std::set<INT> c;
    for (int ix = 0; ix < 19; ix++) {
      c.insert((rank * 17 + 21 + ix) % 137);
    }
    return c;
  };

  std::set<INT> contrib = lambda_get_contrib(rank);

  std::set<INT> correct;
  for (int rx = 0; rx < size; rx++) {
    auto tmp = lambda_get_contrib(rx);
    for (int ix : tmp) {
      correct.insert(ix);
    }
  }

  for (int root = 0; root < size; root++) {
    auto to_test = set_reduce_union(contrib, root, MPI_COMM_WORLD);
    if (rank == root) {
      ASSERT_EQ(to_test, correct);
    } else {
      ASSERT_EQ(to_test, std::set<INT>{});
    }
  }

  auto to_test = set_all_reduce_union(contrib, MPI_COMM_WORLD);
  ASSERT_EQ(correct, to_test);
}

TEST(communication_utility, cart_decomp) {
  std::vector<int> mpi_dims_correct(3);
  for (const int ndim : {1, 2, 3}) {
    mpi_dims_correct.resize(ndim);
    for (const int size : {1, 4, 6, 7, 8, 32, 1024}) {
      std::fill(mpi_dims_correct.begin(), mpi_dims_correct.end(), 0);
      MPICHK(MPI_Dims_create(size, ndim, mpi_dims_correct.data()));
      auto mpi_dims_to_test = get_cart_dims(size, ndim);
      for (int dx = 0; dx < ndim; dx++) {
        ASSERT_EQ(mpi_dims_to_test.at(dx), mpi_dims_correct.at(dx));
      }
    }
  }

  {
    std::vector<int> cell_counts = {100, 200, 300};
    std::vector<int> mpi_dims = {1, 2, 3};
    auto to_test = get_reordered_cart_decomp(3, mpi_dims, cell_counts);
    ASSERT_EQ(to_test.at(0), 1);
    ASSERT_EQ(to_test.at(1), 2);
    ASSERT_EQ(to_test.at(2), 3);
  }

  {
    std::vector<int> cell_counts = {100, 200, 300};
    std::vector<int> mpi_dims = {2, 1, 3};
    auto to_test = get_reordered_cart_decomp(3, mpi_dims, cell_counts);
    ASSERT_EQ(to_test.at(0), 1);
    ASSERT_EQ(to_test.at(1), 2);
    ASSERT_EQ(to_test.at(2), 3);
  }

  {
    std::vector<int> cell_counts = {300, 200, 100};
    std::vector<int> mpi_dims = {2, 1, 3};
    auto to_test = get_reordered_cart_decomp(3, mpi_dims, cell_counts);
    ASSERT_EQ(to_test.at(0), 3);
    ASSERT_EQ(to_test.at(1), 2);
    ASSERT_EQ(to_test.at(2), 1);
  }

  {
    std::vector<int> cell_counts = {1, 1, 32};
    auto tmp = get_lower_dimension_cart_decomp(32, 3, cell_counts);
    ASSERT_EQ(tmp[0], 1);
    ASSERT_EQ(tmp[1], 1);
    ASSERT_EQ(tmp[2], 32);
  }

  {
    std::vector<int> cell_counts = {1, 1, 32};
    auto tmp = get_lower_dimension_cart_decomp(8, 3, cell_counts);
    ASSERT_EQ(tmp[0], 1);
    ASSERT_EQ(tmp[1], 1);
    ASSERT_EQ(tmp[2], 8);
  }
  {
    std::vector<int> cell_counts = {1, 32, 1};
    auto tmp = get_lower_dimension_cart_decomp(8, 3, cell_counts);
    ASSERT_EQ(tmp[0], 1);
    ASSERT_EQ(tmp[1], 8);
    ASSERT_EQ(tmp[2], 1);
  }

  {
    std::vector<int> cell_counts = {1, 2, 32};
    auto tmp = get_lower_dimension_cart_decomp(64, 3, cell_counts);
    ASSERT_EQ(tmp[0], 1);
    ASSERT_EQ(tmp[1], 2);
    ASSERT_EQ(tmp[2], 32);
  }
  {
    std::vector<int> cell_counts = {2, 2, 32};
    auto tmp = get_lower_dimension_cart_decomp(64, 3, cell_counts);
    ASSERT_EQ(tmp[0], 2);
    ASSERT_EQ(tmp[1], 2);
    ASSERT_EQ(tmp[2], 16);
  }

  {
    std::vector<int> cell_counts = {2, 3, 2};
    auto tmp = get_lower_dimension_cart_decomp(12, 3, cell_counts);
    ASSERT_EQ(tmp[0], 2);
    ASSERT_EQ(tmp[1], 3);
    ASSERT_EQ(tmp[2], 2);
  }

  {
    std::vector<int> cell_counts = {1, 1, 128};
    auto tmp = get_lower_dimension_cart_decomp(37, 3, cell_counts);
    ASSERT_EQ(tmp[0], 1);
    ASSERT_EQ(tmp[1], 1);
    ASSERT_EQ(tmp[2], 37);
  }

  {
    std::vector<int> cell_counts = {1, 1, 128};
    auto tmp = get_single_dimension_cart_decomp(37, 3, cell_counts);
    ASSERT_EQ(tmp[0], 1);
    ASSERT_EQ(tmp[1], 1);
    ASSERT_EQ(tmp[2], 37);
  }

  {
    std::vector<int> cell_counts = {3, 7, 127};
    auto tmp = get_single_dimension_cart_decomp(4, 3, cell_counts);
    ASSERT_EQ(tmp[0], 1);
    ASSERT_EQ(tmp[1], 1);
    ASSERT_EQ(tmp[2], 4);
  }

  {
    std::vector<int> cell_counts = {127, 2};
    auto tmp = get_single_dimension_cart_decomp(4, 2, cell_counts);
    ASSERT_EQ(tmp[0], 4);
    ASSERT_EQ(tmp[1], 1);
  }
}
