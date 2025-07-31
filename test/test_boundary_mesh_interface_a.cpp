#include "include/test_neso_particles.hpp"

TEST(BoundaryMeshInterface, mpi_neighbours) {

  MPI_Comm comm = MPI_COMM_WORLD;
  auto sycl_target = std::make_shared<SYCLTarget>(0, comm);
  int rank = -1;
  int size = -1;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  const int num_owned_geoms = 4;

  std::vector<INT> owned_face_cells;
  for (int ix = 0; ix < num_owned_geoms; ix++) {
    owned_face_cells.push_back(rank * num_owned_geoms + ix);
  }

  BoundaryMeshInterface bmi(comm, sycl_target, owned_face_cells);
  const int ncomp = 7;

  ASSERT_EQ(owned_face_cells, bmi.boundary.owned_geom_ids);

  std::map<int, std::vector<std::pair<int, int>>> test_map;
  std::map<int, std::map<int, std::vector<REAL>>> test_data;
  std::map<int, int> test_num_neighbours;

  int correct_num_edges_t = 0;
  std::vector<int> correct_in_degree_t(size);
  std::fill(correct_in_degree_t.begin(), correct_in_degree_t.end(), 0);
  std::vector<int> correct_in_degree(size);
  std::fill(correct_in_degree.begin(), correct_in_degree.end(), 0);

  for (int rankx = 0; rankx < size; rankx++) {
    test_num_neighbours[rankx] = 0;
    int num_neighbours_inner = 0;
    // rank 0 has no outgoing edges
    if (rankx > 0) {
      num_neighbours_inner = rankx + 1;
    }
    for (int nx = 0; nx < num_neighbours_inner; nx++) {
      const int rx = nx % size;

      // rank 3 has no incoming edges
      if (rx != 3) {
        const int gx = rx * num_owned_geoms + (rx + rankx) % num_owned_geoms;
        test_map[rankx].push_back({rx, gx});

        std::vector<REAL> tmp_data(ncomp);
        for (int ix = 0; ix < ncomp; ix++) {
          tmp_data[ix] = gx * 0.31245 + std::pow(rx, ix) * 0.123;
        }

        test_data[rankx][gx] = tmp_data;

        if (rankx == rank) {
          correct_in_degree_t.at(rx)++;
          correct_num_edges_t++;
        }
        test_num_neighbours[rankx]++;
      }
    }
  }

  int correct_num_edges = 0;
  MPICHK(MPI_Allreduce(&correct_num_edges_t, &correct_num_edges, 1, MPI_INT,
                       MPI_SUM, comm));

  MPICHK(MPI_Allreduce(correct_in_degree_t.data(), correct_in_degree.data(),
                       size, MPI_INT, MPI_SUM, comm));

  bmi.extend_exchange_pattern(test_map[rank]);

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
  ASSERT_EQ(outdegree, test_num_neighbours[rank]);
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

  const auto incoming_geom_ids_size = bmi.boundary.incoming_geom_ids.size();
  std::vector<int> incoming_geoms_to_test_v(incoming_geom_ids_size);

  if (incoming_geom_ids_size > 0) {
    sycl_target->queue
        .memcpy(incoming_geoms_to_test_v.data(),
                bmi.boundary.d_incoming_geom_ids->ptr,
                incoming_geom_ids_size * sizeof(int))
        .wait_and_throw();
  }
  incoming_geoms_to_test.clear();
  for (auto ix : incoming_geoms_to_test_v) {
    incoming_geoms_to_test.insert(ix);
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

  std::vector<REAL> outdata;
  outdata.reserve(bmi.boundary.total_num_outgoing_geoms);
  for (int gx : bmi.boundary.outgoing_geom_ids) {
    outdata.insert(outdata.end(), test_data.at(rank).at(gx).begin(),
                   test_data.at(rank).at(gx).end());
  }

  std::vector<REAL> indata(bmi.boundary.total_num_incoming_geoms * ncomp);
  std::fill(indata.begin(), indata.end(), -1.0);

  bmi.exchange_surface(outdata.data(), ncomp, indata.data());

  std::vector<REAL> indata_correct;

  indata_correct.reserve(bmi.boundary.total_num_incoming_geoms * ncomp);

  for (int source_rank : bmi.boundary.graph.sources) {
    for (int gid : bmi.boundary.map_send_rank_to_geom_ids.at(source_rank)) {
      auto &tmp = test_data.at(source_rank).at(gid);
      indata_correct.insert(indata_correct.end(), tmp.begin(), tmp.end());
    }
  }

  ASSERT_EQ(indata, indata_correct);

  // Test the DOF packing loop
  {
    const std::size_t packed_data_length =
        bmi.boundary.outgoing_geom_ids.size() * ncomp;

    std::vector<REAL> h_src(packed_data_length);
    std::vector<REAL> h_dst(packed_data_length);
    std::fill(h_dst.begin(), h_dst.end(), 0.0);
    for (std::size_t ix = 0; ix < packed_data_length; ix++) {
      h_src[ix] = ix * 0.01;
    }
    BufferDevice<REAL> d_src(sycl_target, h_src);
    BufferDevice<REAL> d_dst(sycl_target, h_dst);

    bmi.exchange_from_device_pack(d_src.ptr, ncomp, d_dst.ptr).wait_and_throw();

    sycl_target->queue
        .memcpy(h_dst.data(), d_dst.ptr, packed_data_length * sizeof(REAL))
        .wait_and_throw();

    int index_dst = 0;
    for (int gx : bmi.boundary.outgoing_geom_ids) {
      const int index_src = bmi.boundary.map_geom_id_to_linear_index.at(gx);
      for (int cx = 0; cx < ncomp; cx++) {
        ASSERT_EQ(h_dst[index_dst * ncomp + cx], h_src[index_src * ncomp + cx]);
      }
      index_dst++;
    }
  }
  // Test the DOF unpacking loop
  {
    const std::size_t packed_data_length_src =
        bmi.boundary.incoming_geom_ids.size() * ncomp;
    const std::size_t packed_data_length_dst =
        bmi.boundary.owned_geom_ids.size() * ncomp;
    ASSERT_EQ(bmi.boundary.owned_geom_ids.size(), num_owned_geoms);

    if (packed_data_length_src > 0) {
      std::vector<REAL> h_src(packed_data_length_src);
      std::vector<REAL> h_dst(packed_data_length_dst);
      std::fill(h_dst.begin(), h_dst.end(), 0.0);
      for (std::size_t ix = 0; ix < packed_data_length_src; ix++) {
        h_src[ix] = ix * 0.01;
      }
      BufferDevice<REAL> d_src(sycl_target, h_src);
      BufferDevice<REAL> d_dst(sycl_target, h_dst);

      bmi.exchange_from_device_unpack(d_src.ptr, ncomp, d_dst.ptr)
          .wait_and_throw();

      sycl_target->queue
          .memcpy(h_dst.data(), d_dst.ptr,
                  packed_data_length_dst * sizeof(REAL))
          .wait_and_throw();

      int dst_index = 0;
      for (auto gx : owned_face_cells) {
        for (int cx = 0; cx < ncomp; cx++) {
          REAL correct = 0.0;

          int src_index = 0;
          for (auto hx : bmi.boundary.incoming_geom_ids) {
            if (hx == gx) {
              correct += h_src.at(src_index * ncomp + cx);
            }
            src_index++;
          }
          ASSERT_NEAR(h_dst.at(dst_index * ncomp + cx), correct, 1.0e-14);
        }
        dst_index++;
      }
    }
  }

  bmi.free();
  sycl_target->free();
}
