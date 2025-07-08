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

    int rank = -1;
    MPICHK(MPI_Comm_rank(this->boundary.comm, &rank));
    int degrees =
        static_cast<int>(this->boundary.map_recv_rank_to_geom_ids.size());

    std::vector<int> destinations;
    destinations.reserve(degrees);
    for (auto &rx : this->boundary.map_recv_rank_to_geom_ids) {
      destinations.push_back(rx.first);
    }

    MPICHK(MPI_Dist_graph_create(this->boundary.comm, 1, &rank, &degrees,
                                 destinations.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, 0, &this->boundary.ncomm));

    NESOASSERT(this->boundary.ncomm != MPI_COMM_NULL,
               "Failure to setup MPI graph topology.");
  }
}

} // namespace NESO::Particles
