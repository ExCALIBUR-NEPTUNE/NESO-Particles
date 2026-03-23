#include <neso_particles/boundary/boundary_mesh_interface.hpp>
#include <numeric>

namespace NESO::Particles {

BoundaryMeshInterface::BoundaryMeshInterface(
    MPI_Comm comm, SYCLTargetSharedPtr sycl_target,
    const std::vector<INT> &owned_face_cells) {
  this->comm = comm;
  this->sycl_target = sycl_target;

  MPICHK(MPI_Dist_graph_create(this->comm, 0, nullptr, nullptr, nullptr,
                               MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &this->ncomm));
  NESOASSERT(this->ncomm != MPI_COMM_NULL,
             "Failure to setup MPI graph topology.");

  this->d_map_geom_id_to_linear_index =
      std::make_shared<BlockedBinaryTree<INT, INT>>(sycl_target);

  this->d_outgoing_pack_index =
      std::make_shared<BufferDevice<int>>(sycl_target, 8);
  this->device_aware_mpi = device_aware_mpi_enabled();

  this->owned_geom_ids = owned_face_cells;
  this->d_map_owned_geom_id_to_linear_index =
      std::make_shared<BlockedBinaryTree<INT, INT>>(sycl_target);

  INT owned_linear_index = 0;
  for (const INT &gid : owned_face_cells) {
    this->map_owned_geom_id_to_linear_index[gid] = owned_linear_index;
    this->d_map_owned_geom_id_to_linear_index->add(gid, owned_linear_index);
    owned_linear_index++;
  }
}

INT BoundaryMeshInterface::get_num_intersection_geoms() const {
  return this->geom_counter;
}

INT BoundaryMeshInterface::get_total_num_exported_geoms() const {
  return static_cast<INT>(this->d_reverse_outgoing_pack_index->size);
}

void BoundaryMeshInterface::free() {
  if (this->ncomm != MPI_COMM_NULL) {
    MPICHK(MPI_Comm_free(&this->ncomm));
    this->ncomm = MPI_COMM_NULL;
  }
  if (this->rncomm != MPI_COMM_NULL) {
    MPICHK(MPI_Comm_free(&this->rncomm));
    this->rncomm = MPI_COMM_NULL;
  }
}

void BoundaryMeshInterface::extend_exchange_pattern(
    const std::vector<std::pair<int, INT>> &rank_geom_ids) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "BoundaryMeshInterface", "extend_exchange_pattern");

  NESOASSERT(this->comm != MPI_COMM_NULL,
             "BoundaryMeshInterface::boundary_init has not been called.");

  int map_is_modified_t = 0;
  for (auto &[rank, geom_id] : rank_geom_ids) {
    // Is this rank, geom_id pair new?
    if (!this->map_recv_rank_to_geom_ids[rank].count(geom_id)) {
      map_is_modified_t = 1;
      this->map_recv_rank_to_geom_ids[rank].insert(geom_id);
      const INT linear_seqential_index = this->geom_counter++;
      this->map_linear_index_to_geom_id[linear_seqential_index] = geom_id;
      this->map_geom_id_to_linear_index[geom_id] = linear_seqential_index;
      this->d_map_geom_id_to_linear_index->add(geom_id, linear_seqential_index);
      this->extended_pattern_geom_ids.insert(geom_id);
    }
  }

  int map_is_modified = 0;
  MPICHK(MPI_Allreduce(&map_is_modified_t, &map_is_modified, 1, MPI_INT,
                       MPI_MAX, comm));

  // If no rank actually has any new geoms to inform the owner about then there
  // is nothing to do.
  if (map_is_modified) {
    if (this->ncomm != MPI_COMM_NULL) {
      MPICHK(MPI_Comm_free(&this->ncomm));
    }
    this->map_typencomp_alltoallwargs.clear();
    this->reverse_map_typencomp_alltoallwargs.clear();

    // Create a distributed graph with the new topology.
    int rank = -1;
    MPICHK(MPI_Comm_rank(this->comm, &rank));
    int degrees = static_cast<int>(this->map_recv_rank_to_geom_ids.size());

    std::vector<int> destinations;
    destinations.reserve(degrees);
    std::vector<int> owned_geom_counts;
    owned_geom_counts.reserve(degrees);
    for (auto &rx : this->map_recv_rank_to_geom_ids) {
      destinations.push_back(rx.first);
      owned_geom_counts.push_back(static_cast<int>(rx.second.size()));
    }

    // get a legitimate pointer to avoid the mpi implementations complaining
    // about nullptrs on arrays they don't access....
    int dummy_destinations = 1;

    // Graph for the projection direction
    MPICHK(MPI_Dist_graph_create(
        this->comm, 1, &rank, &degrees,
        destinations.size() ? destinations.data() : &dummy_destinations,
        MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &this->ncomm));
    NESOASSERT(this->ncomm != MPI_COMM_NULL,
               "Failure to setup MPI graph topology.");

    // Graph for the evaluation direction
    if (this->rncomm != MPI_COMM_NULL) {
      MPICHK(MPI_Comm_free(&this->rncomm));
    }
    MPICHK(
        reverse_graph_edge_directions(this->comm, this->ncomm, &this->rncomm));
    NESOASSERT(this->rncomm != MPI_COMM_NULL,
               "Failure to setup MPI graph topology (reverse direction).");

    // Extract the topology from the graph that was created.
    this->graph.indegree = -1;
    this->graph.outdegree = -1;
    int weighted = -1;
    MPICHK(MPI_Dist_graph_neighbors_count(this->ncomm, &this->graph.indegree,
                                          &this->graph.outdegree, &weighted));
    NESOASSERT(this->graph.outdegree == degrees,
               "Unexpected number of out edges.");
    NESOASSERT(weighted == 0, "Expected unweighted.");

    std::vector<int> sourcesweights(this->graph.indegree);
    std::vector<int> destweights(this->graph.outdegree);

    this->graph.sources.resize(this->graph.indegree);
    this->graph.destinations.resize(this->graph.outdegree);

    MPICHK(MPI_Dist_graph_neighbors(
        this->ncomm, this->graph.indegree, this->graph.sources.data(),
        sourcesweights.data(), this->graph.outdegree,
        this->graph.destinations.data(), destweights.data()));

    // Now we have the topology, exchange to the owning ranks how many geoms
    // this rank holds copies of for that rank.
    this->incoming_geom_counts.resize(this->graph.indegree);
    std::fill(this->incoming_geom_counts.begin(),
              this->incoming_geom_counts.end(), 0);

    // We have to reorder these array as the outward edges in the MPI
    // representation of the graph might be different to the order in the map.
    this->outgoing_geom_counts.resize(this->graph.outdegree);
    this->total_num_outgoing_geoms = 0;
    for (int dst_rank_index = 0; dst_rank_index < this->graph.outdegree;
         dst_rank_index++) {
      const int dst_rank = this->graph.destinations.at(dst_rank_index);
      const int tmp_count = this->map_recv_rank_to_geom_ids[dst_rank].size();
      this->outgoing_geom_counts.at(dst_rank_index) = tmp_count;
      this->total_num_outgoing_geoms += tmp_count;
    }

    // mpich complains that the input pointers are nullptr even if that data is
    // not accessed.
    int null_out = -1;
    int null_in = -1;
    int *out_data_counts = this->outgoing_geom_counts.size()
                               ? this->outgoing_geom_counts.data()
                               : &null_out;
    int *in_data_counts = this->incoming_geom_counts.size()
                              ? this->incoming_geom_counts.data()
                              : &null_in;

    MPICHK(NP_MPI_Neighbor_alltoall_wrapper(
        out_data_counts, 1, MPI_INT, in_data_counts, 1, MPI_INT, this->ncomm));

    // Send the geometry id to the corresponding owning rank such that the
    // owning rank knows which geometry object incoming data corresonds to.
    this->total_num_incoming_geoms =
        std::accumulate(this->incoming_geom_counts.begin(),
                        this->incoming_geom_counts.end(), 0);

    // Realloc the incoming ids vector
    this->incoming_geom_ids.resize(this->total_num_incoming_geoms);
    // Realloc the outgoing ids vector
    this->outgoing_geom_ids.resize(this->total_num_outgoing_geoms);
    // Populate the outgoing ids vector in the order that MPI has the edges in
    // the graph.
    int index = 0;
    for (int dst_rank_index = 0; dst_rank_index < this->graph.outdegree;
         dst_rank_index++) {
      const int dst_rank = this->graph.destinations.at(dst_rank_index);
      for (int gx : this->map_recv_rank_to_geom_ids[dst_rank]) {
        this->outgoing_geom_ids.at(index++) = gx;
      }
    }
    NESOASSERT(index == this->total_num_outgoing_geoms,
               "Bookkeeping error in indexing.");

    this->exchange_surface(this->outgoing_geom_ids.data(), 1,
                           this->incoming_geom_ids.data());

    // Create the device side copy of the incoming ids.
    this->d_incoming_geom_ids = std::make_shared<BufferDevice<int>>(
        this->sycl_target, this->incoming_geom_ids);

    // populate map_send_rank_to_geom_ids
    index = 0;
    for (int src_rank_index = 0; src_rank_index < this->graph.indegree;
         src_rank_index++) {
      const int src_rank = this->graph.sources[src_rank_index];
      for (int ix = 0; ix < this->incoming_geom_counts[src_rank_index]; ix++) {
        const int gid = this->incoming_geom_ids[index++];
        this->map_send_rank_to_geom_ids[src_rank].insert(gid);

        NESOASSERT(this->map_owned_geom_id_to_linear_index.count(gid),
                   "This MPI rank is being sent information for a geom it does "
                   "not own.");
      }
    }

    // (Re)Create the device packing indices
    this->d_outgoing_pack_index->realloc_no_copy(this->geom_counter);
    std::vector<int> h_outgoing_pack_index(this->geom_counter);
    std::fill(h_outgoing_pack_index.begin(), h_outgoing_pack_index.end(), -1);
    int pack_index = 0;
    for (auto outgoing_geom : this->outgoing_geom_ids) {
      const int seq_linear_index =
          this->map_geom_id_to_linear_index.at(outgoing_geom);
      h_outgoing_pack_index.at(seq_linear_index) = pack_index++;
    }
    auto e0 = this->sycl_target->queue.memcpy(this->d_outgoing_pack_index->ptr,
                                              h_outgoing_pack_index.data(),
                                              this->geom_counter * sizeof(int));

    for (int ix = 0; ix < this->geom_counter; ix++) {
      NESOASSERT(h_outgoing_pack_index[ix] != -1,
                 "Expected all entries to be populated.");
    }

    // Create the packing maps for the evaluation direction. We have to assume
    // that MPI may have reordered the edges for each node in the graph.
    this->graph.reverse_sources.resize(this->graph.outdegree);
    this->graph.reverse_destinations.resize(this->graph.indegree);
    MPICHK(MPI_Dist_graph_neighbors(
        this->rncomm, this->graph.outdegree, this->graph.reverse_sources.data(),
        destweights.data(), this->graph.indegree,
        this->graph.reverse_destinations.data(), sourcesweights.data()));

    // Create the list of (linear) geom indices to pack in the order that they
    // should be passed to MPI.
    std::vector<int> eval_pack_indices;
    eval_pack_indices.reserve(this->incoming_geom_ids.size());

    index = 0;
    this->reverse_outgoing_geom_counts.resize(
        this->graph.reverse_destinations.size());
    for (const int dst_rank : this->graph.reverse_destinations) {
      for (const int gid : this->map_send_rank_to_geom_ids.at(dst_rank)) {
        // Get the linear source index from the geom id
        const auto linear_index =
            this->map_owned_geom_id_to_linear_index.at(gid);
        eval_pack_indices.push_back(static_cast<int>(linear_index));
      }
      this->reverse_outgoing_geom_counts[index] =
          this->map_send_rank_to_geom_ids.at(dst_rank).size();
      index++;
    }
    this->d_reverse_outgoing_pack_index = std::make_shared<BufferDevice<int>>(
        this->sycl_target, eval_pack_indices);

    // Create the map from the index DOFs were received on to the linear index
    // for evaluation.
    std::vector<int> eval_unpack_indices;
    eval_unpack_indices.reserve(this->outgoing_geom_ids.size());

    index = 0;
    this->reverse_incoming_geom_counts.resize(
        this->graph.reverse_sources.size());
    for (const int src_rank : this->graph.reverse_sources) {
      for (const int gid : this->map_recv_rank_to_geom_ids.at(src_rank)) {
        const auto linear_index = this->map_geom_id_to_linear_index.at(gid);
        eval_unpack_indices.push_back(static_cast<int>(linear_index));
      }
      this->reverse_incoming_geom_counts[index] =
          this->map_recv_rank_to_geom_ids.at(src_rank).size();
      index++;
    }
    this->d_reverse_incoming_unpack_index = std::make_shared<BufferDevice<int>>(
        this->sycl_target, eval_unpack_indices);

    NESOASSERT(this->reverse_incoming_geom_counts.size() ==
                   this->outgoing_geom_counts.size(),
               "Missmatch in bookkeeping array sizes.");

    NESOASSERT(this->reverse_outgoing_geom_counts.size() ==
                   this->incoming_geom_counts.size(),
               "Missmatch in bookkeeping array sizes.");

    e0.wait_and_throw();

    // As the exchange maps have been updated then update the version.
    this->version++;
  }

  this->sycl_target->profile_map.end_region(r0);
}

INT BoundaryMeshInterface::get_geom_id_from_seq_index(
    const INT linear_seq_index) {
  return this->map_linear_index_to_geom_id.at(linear_seq_index);
}

INT BoundaryMeshInterface::get_seq_index_from_geom_id(const INT geom_id) {
  return this->map_geom_id_to_linear_index.at(geom_id);
}

std::tuple<
    BlockedBinaryNode<INT, INT, NESO_PARTICLES_BLOCKED_BINARY_TREE_WIDTH> *,
    INT>
BoundaryMeshInterface::get_device_geom_id_to_seq() {

  return {this->d_map_geom_id_to_linear_index->root, this->geom_counter};
}

std::function<std::int64_t()>
BoundaryMeshInterface::get_version_function_handle() {

  return [&]() -> std::int64_t { return this->version; };
}

std::set<INT> BoundaryMeshInterface::get_extended_pattern_geom_ids() {
  return this->extended_pattern_geom_ids;
}

void BoundaryMeshInterface::print_reverse_info() {

  const int rank = this->sycl_target->comm_pair.rank_parent;
  const int size = this->sycl_target->comm_pair.size_parent;

  for (int rankx = 0; rankx < size; rankx++) {
    if (rankx == rank) {
      nprint("rank:", rank);
      nprint("\tSENDING:");
      const int num_destinations =
          static_cast<int>(this->graph.reverse_destinations.size());

      if (num_destinations) {
        auto h_reverse_outgoing_pack_index =
            this->d_reverse_outgoing_pack_index->get();

        int pack_index_dst = 0;
        for (int dx = 0; dx < num_destinations; dx++) {
          const int dest_rank = this->graph.reverse_destinations.at(dx);
          const int geom_count = this->reverse_outgoing_geom_counts[dx];
          nprint("\t\t", "destination rank:", dest_rank,
                 "num geoms:", geom_count);
          for (int gx = 0; gx < geom_count; gx++) {
            const auto source_index =
                h_reverse_outgoing_pack_index.at(pack_index_dst);
            const INT geom_id = this->owned_geom_ids.at(source_index);
            nprint("\t\t\t", "pack index:", pack_index_dst,
                   "source index:", source_index, "geom id:", geom_id);
            pack_index_dst++;
          }
        }
      }

      nprint("\tRECVING:");
      const int num_sources =
          static_cast<int>(this->graph.reverse_sources.size());
      if (num_sources) {

        auto h_reverse_incoming_unpack_index =
            d_reverse_incoming_unpack_index->get();

        int pack_index_src = 0;
        for (int sx = 0; sx < num_sources; sx++) {
          const int src_rank = this->graph.reverse_sources.at(sx);
          const int geom_count = reverse_incoming_geom_counts.at(sx);
          nprint("\t\t", "source rank:", src_rank, "num geoms:", geom_count);
          for (int gx = 0; gx < geom_count; gx++) {
            const auto dst_index =
                h_reverse_incoming_unpack_index[pack_index_src];
            const INT geom_id = map_linear_index_to_geom_id.at(dst_index);

            nprint("\t\t\t", "src index:", pack_index_src,
                   "linear_stage_index:", dst_index, "geom id:", geom_id);
            pack_index_src++;
          }
        }
      }

      std::cout << std::flush;
    }
    MPICHK(MPI_Barrier(this->sycl_target->comm_pair.comm_parent));
    std::cout << std::flush;
    MPICHK(MPI_Barrier(this->sycl_target->comm_pair.comm_parent));
  }
}

template void BoundaryMeshInterface::exchange_surface(REAL *data,
                                                      const int ncomp,
                                                      REAL *data_gathered);
template void BoundaryMeshInterface::exchange_from_device(REAL *d_src,
                                                          const int ncomp,
                                                          REAL *d_dst);

template void
BoundaryMeshInterface::reverse_exchange_surface(REAL *data, const int ncomp,
                                                REAL *data_gathered);
template void BoundaryMeshInterface::reverse_exchange_from_device(
    REAL *d_src, const int ncomp, REAL *d_dst);

template sycl::event
BoundaryMeshInterface::exchange_from_device_pack(REAL *k_packed_in,
                                                 const int ncomp, REAL *d_dst);

template sycl::event BoundaryMeshInterface::exchange_from_device_unpack(
    REAL *k_packed_in, const int ncomp, REAL *d_dst);

template sycl::event BoundaryMeshInterface::reverse_exchange_from_device_pack(
    REAL *k_packed_in, const int ncomp, REAL *d_dst);

template sycl::event BoundaryMeshInterface::reverse_exchange_from_device_unpack(
    REAL *k_packed_in, const int ncomp, REAL *d_dst);

} // namespace NESO::Particles
