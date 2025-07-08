#include <neso_particles/boundary/boundary_mesh_interface.hpp>

namespace NESO::Particles {

void BoundaryMeshInterface::boundary_init(MPI_Comm comm) {
  this->boundary.comm = comm;

  MPICHK(MPI_Dist_graph_create(this->boundary.comm, 0, nullptr, nullptr,
                               nullptr, MPI_UNWEIGHTED, MPI_INFO_NULL, 0,
                               &this->boundary.ncomm));

  NESOASSERT(this->boundary.ncomm != MPI_COMM_NULL,
             "Failure to setup MPI graph topology.");
}

void BoundaryMeshInterface::boundary_free() {
  MPICHK(MPI_Comm_free(&this->boundary.ncomm));
}

void BoundaryMeshInterface::boundary_extend_exchange_pattern(
    const std::vector<std::pair<int, int>> &rank_geom_ids) {

  MPI_Comm comm = this->boundary.comm;
  NESOASSERT(comm != MPI_COMM_NULL,
             "BoundaryMeshInterface::boundary_init has not been called.");
  NESOASSERT(this->boundary.comm != MPI_COMM_NULL,
             "BoundaryMeshInterface::boundary_init has not been called.");

  std::map<int, std::set<int>> map_recv_rank_to_new_geom_ids;

  int map_is_modified_t = 0;
  for (auto &[rank, geom_id] : rank_geom_ids) {
    // Is this rank, geom_id pair new?
    if (!this->boundary.map_recv_rank_to_geom_ids[rank].count(geom_id)) {
      map_is_modified_t = 1;
      this->boundary.map_recv_rank_to_geom_ids[rank].insert(geom_id);
    }
  }

  int map_is_modified = 0;
  MPICHK(MPI_Allreduce(&map_is_modified_t, &map_is_modified, 1, MPI_INT,
                       MPI_MAX, comm));

  nprint_variable(map_is_modified);
  // If no rank actually has any new geoms to inform the owner about then there
  // is nothing to do.
  if (map_is_modified) {
    MPICHK(MPI_Comm_free(&this->boundary.ncomm));

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

    nprint_variable(rank);
    nprint_variable(this->boundary.graph.outdegree);
    nprint_variable(this->boundary.graph.indegree);

    // We have to reorder these array as the outward edges in the MPI
    // representation of the graph might be different to the order in the map.
    std::vector<int> outgoing_geom_counts(this->boundary.graph.outdegree);
    for (int dst_rank_index = 0;
         dst_rank_index < this->boundary.graph.outdegree; dst_rank_index++) {
      const int dst_rank = this->boundary.graph.destinations.at(dst_rank_index);
      outgoing_geom_counts.at(dst_rank_index) =
          this->boundary.map_recv_rank_to_geom_ids[dst_rank].size();
    }

    // mpich complains that the input pointers are nullptr even if that data is
    // not accessed.
    int null_out = -1;
    int null_in = -1;
    int *out_data =
        outgoing_geom_counts.size() ? outgoing_geom_counts.data() : &null_out;
    int *in_data = this->boundary.incoming_geom_counts.size()
                       ? this->boundary.incoming_geom_counts.data()
                       : &null_in;
    MPICHK(MPI_Neighbor_allgather(out_data, 1, MPI_INT, in_data, 1, MPI_INT,
                                  this->boundary.ncomm));
  }
}

} // namespace NESO::Particles
