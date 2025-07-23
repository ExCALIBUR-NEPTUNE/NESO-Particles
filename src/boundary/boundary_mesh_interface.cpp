#include <neso_particles/boundary/boundary_mesh_interface.hpp>
#include <numeric>

namespace NESO::Particles {

BoundaryMeshInterface::BoundaryMeshInterface(MPI_Comm comm,
                                             SYCLTargetSharedPtr sycl_target) {
  this->boundary.comm = comm;
  this->boundary.sycl_target = sycl_target;
  this->boundary.d_map_geom_id_to_linear_index =
      std::make_shared<BlockedBinaryTree<INT, INT>>(sycl_target);

  MPICHK(MPI_Dist_graph_create(this->boundary.comm, 0, nullptr, nullptr,
                               nullptr, MPI_UNWEIGHTED, MPI_INFO_NULL, 0,
                               &this->boundary.ncomm));

  NESOASSERT(this->boundary.ncomm != MPI_COMM_NULL,
             "Failure to setup MPI graph topology.");

  this->boundary.d_outgoing_pack_index =
      std::make_shared<BufferDevice<int>>(sycl_target, 8);
  this->boundary.device_aware_mpi = device_aware_mpi_enabled();
}

void BoundaryMeshInterface::free() {
  MPICHK(MPI_Comm_free(&this->boundary.ncomm));
}

void BoundaryMeshInterface::extend_exchange_pattern(
    const std::vector<std::pair<int, int>> &rank_geom_ids) {

  MPI_Comm comm = this->boundary.comm;
  NESOASSERT(comm != MPI_COMM_NULL,
             "BoundaryMeshInterface::boundary_init has not been called.");
  NESOASSERT(this->boundary.comm != MPI_COMM_NULL,
             "BoundaryMeshInterface::boundary_init has not been called.");

  int map_is_modified_t = 0;
  for (auto &[rank, geom_id] : rank_geom_ids) {
    // Is this rank, geom_id pair new?
    if (!this->boundary.map_recv_rank_to_geom_ids[rank].count(geom_id)) {
      map_is_modified_t = 1;
      this->boundary.map_recv_rank_to_geom_ids[rank].insert(geom_id);
      const INT linear_seqential_index = this->boundary.geom_counter++;
      this->boundary.map_linear_index_to_geom_id[linear_seqential_index] =
          geom_id;
      this->boundary.map_geom_id_to_linear_index[geom_id] =
          linear_seqential_index;
      this->boundary.d_map_geom_id_to_linear_index->add(geom_id,
                                                        linear_seqential_index);
    }
  }

  int map_is_modified = 0;
  MPICHK(MPI_Allreduce(&map_is_modified_t, &map_is_modified, 1, MPI_INT,
                       MPI_MAX, comm));

  // If no rank actually has any new geoms to inform the owner about then there
  // is nothing to do.
  if (map_is_modified) {
    MPICHK(MPI_Comm_free(&this->boundary.ncomm));
    this->boundary.map_typencomp_alltoallwargs.clear();

    // Create a distributed graph with the new topology.
    int rank = -1;
    MPICHK(MPI_Comm_rank(this->boundary.comm, &rank));
    int degrees =
        static_cast<int>(this->boundary.map_recv_rank_to_geom_ids.size());

    std::vector<int> destinations;
    destinations.reserve(degrees);
    std::vector<int> owned_geom_counts;
    owned_geom_counts.reserve(degrees);
    for (auto &rx : this->boundary.map_recv_rank_to_geom_ids) {
      destinations.push_back(rx.first);
      owned_geom_counts.push_back(static_cast<int>(rx.second.size()));
    }

    MPICHK(MPI_Dist_graph_create(this->boundary.comm, 1, &rank, &degrees,
                                 destinations.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, 0, &this->boundary.ncomm));

    NESOASSERT(this->boundary.ncomm != MPI_COMM_NULL,
               "Failure to setup MPI graph topology.");

    // Extract the topology from the graph that was created.
    this->boundary.graph.indegree = -1;
    this->boundary.graph.outdegree = -1;
    int weighted = -1;
    MPICHK(MPI_Dist_graph_neighbors_count(
        this->boundary.ncomm, &this->boundary.graph.indegree,
        &this->boundary.graph.outdegree, &weighted));
    NESOASSERT(this->boundary.graph.outdegree == degrees,
               "Unexpected number of out edges.");
    NESOASSERT(weighted == 0, "Expected unweighted.");

    std::vector<int> sourcesweights(this->boundary.graph.indegree);
    std::vector<int> destweights(this->boundary.graph.outdegree);

    this->boundary.graph.sources.resize(this->boundary.graph.indegree);
    this->boundary.graph.destinations.resize(this->boundary.graph.outdegree);

    MPICHK(MPI_Dist_graph_neighbors(
        this->boundary.ncomm, this->boundary.graph.indegree,
        this->boundary.graph.sources.data(), sourcesweights.data(),
        this->boundary.graph.outdegree,
        this->boundary.graph.destinations.data(), destweights.data()));

    // Now we have the topology, exchange to the owning ranks how many geoms
    // this rank holds copies of for that rank.
    this->boundary.incoming_geom_counts.resize(this->boundary.graph.indegree);
    std::fill(this->boundary.incoming_geom_counts.begin(),
              this->boundary.incoming_geom_counts.end(), 0);

    // We have to reorder these array as the outward edges in the MPI
    // representation of the graph might be different to the order in the map.
    this->boundary.outgoing_geom_counts.resize(this->boundary.graph.outdegree);
    this->boundary.total_num_outgoing_geoms = 0;
    for (int dst_rank_index = 0;
         dst_rank_index < this->boundary.graph.outdegree; dst_rank_index++) {
      const int dst_rank = this->boundary.graph.destinations.at(dst_rank_index);
      const int tmp_count =
          this->boundary.map_recv_rank_to_geom_ids[dst_rank].size();
      this->boundary.outgoing_geom_counts.at(dst_rank_index) = tmp_count;
      this->boundary.total_num_outgoing_geoms += tmp_count;
    }

    // mpich complains that the input pointers are nullptr even if that data is
    // not accessed.
    int null_out = -1;
    int null_in = -1;
    int *out_data_counts = this->boundary.outgoing_geom_counts.size()
                               ? this->boundary.outgoing_geom_counts.data()
                               : &null_out;
    int *in_data_counts = this->boundary.incoming_geom_counts.size()
                              ? this->boundary.incoming_geom_counts.data()
                              : &null_in;
    MPICHK(MPI_Neighbor_allgather(out_data_counts, 1, MPI_INT, in_data_counts,
                                  1, MPI_INT, this->boundary.ncomm));

    // Send the geometry id to the corresponding owning rank such that the
    // owning rank knows which geometry object incoming data corresonds to.
    this->boundary.total_num_incoming_geoms =
        std::accumulate(this->boundary.incoming_geom_counts.begin(),
                        this->boundary.incoming_geom_counts.end(), 0);

    // Realloc the incoming ids vector
    this->boundary.incoming_geom_ids.resize(
        this->boundary.total_num_incoming_geoms);
    // Realloc the outgoing ids vector
    this->boundary.outgoing_geom_ids.resize(
        this->boundary.total_num_outgoing_geoms);
    // Populate the outgoing ids vector in the order that MPI has the edges in
    // the graph.
    int index = 0;
    for (int dst_rank_index = 0;
         dst_rank_index < this->boundary.graph.outdegree; dst_rank_index++) {
      const int dst_rank = this->boundary.graph.destinations.at(dst_rank_index);
      for (int gx : this->boundary.map_recv_rank_to_geom_ids[dst_rank]) {
        this->boundary.outgoing_geom_ids.at(index++) = gx;
      }
    }
    NESOASSERT(index == this->boundary.total_num_outgoing_geoms,
               "Bookkeeping error in indexing.");

    this->exchange_surface(this->boundary.outgoing_geom_ids.data(), 1,
                           this->boundary.incoming_geom_ids.data());

    // populate map_send_rank_to_geom_ids (this is mainly for testing/debugging)
    index = 0;
    for (int src_rank_index = 0; src_rank_index < this->boundary.graph.indegree;
         src_rank_index++) {
      const int src_rank = this->boundary.graph.sources[src_rank_index];
      for (int ix = 0; ix < this->boundary.incoming_geom_counts[src_rank_index];
           ix++) {
        const int gid = this->boundary.incoming_geom_ids[index++];
        this->boundary.map_send_rank_to_geom_ids[src_rank].insert(gid);
      }
    }

    // (Re)Create the device packing indices
    this->boundary.d_outgoing_pack_index->realloc_no_copy(
        this->boundary.geom_counter);
    std::vector<int> h_outgoing_pack_index(this->boundary.geom_counter);
    std::fill(h_outgoing_pack_index.begin(), h_outgoing_pack_index.end(), -1);
    int pack_index = 0;
    for (auto outgoing_geom : this->boundary.outgoing_geom_ids) {
      const int seq_linear_index =
          this->boundary.map_geom_id_to_linear_index.at(outgoing_geom);
      h_outgoing_pack_index.at(seq_linear_index) = pack_index++;
    }
    auto e0 = this->boundary.sycl_target->queue.memcpy(
        this->boundary.d_outgoing_pack_index->ptr, h_outgoing_pack_index.data(),
        this->boundary.geom_counter * sizeof(int));

    for (int ix = 0; ix < this->boundary.geom_counter; ix++) {
      NESOASSERT(h_outgoing_pack_index[ix] != -1,
                 "Expected all entries to be populated.");
    }

    e0.wait_and_throw();
  }
}

INT BoundaryMeshInterface::get_geom_id_from_seq_index(
    const INT linear_seq_index) {
  return this->boundary.map_linear_index_to_geom_id.at(linear_seq_index);
}

INT BoundaryMeshInterface::get_seq_index_from_geom_id(const INT geom_id) {
  return this->boundary.map_geom_id_to_linear_index.at(geom_id);
}

std::tuple<
    BlockedBinaryNode<INT, INT, NESO_PARTICLES_BLOCKED_BINARY_TREE_WIDTH> *,
    INT>
BoundaryMeshInterface::get_device_geom_id_to_seq() {

  return {this->boundary.d_map_geom_id_to_linear_index->root,
          this->boundary.geom_counter};
}

template void BoundaryMeshInterface::exchange_surface(int *data,
                                                      const int ncomp,
                                                      int *data_gathered);
template void BoundaryMeshInterface::exchange_surface(INT *data,
                                                      const int ncomp,
                                                      INT *data_gathered);
template void BoundaryMeshInterface::exchange_surface(REAL *data,
                                                      const int ncomp,
                                                      REAL *data_gathered);

} // namespace NESO::Particles
