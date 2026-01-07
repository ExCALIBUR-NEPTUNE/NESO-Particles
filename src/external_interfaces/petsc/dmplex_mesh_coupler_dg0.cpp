#ifdef NESO_PARTICLES_PETSC
#include <algorithm>
#include <memory>
#include <neso_particles/external_interfaces/petsc/dmplex_helper.hpp>
#include <neso_particles/external_interfaces/petsc/dmplex_mesh_coupler_dg0.hpp>

namespace NESO::Particles::PetscInterface {

DMPlexMeshCouplerDG0::DMPlexMeshCouplerDG0(
    DM dmplex_B,
    std::vector<std::vector<DMPlexMeshCouplerDG0MapEntry>> &coupling_map)
    : dmplex_B(dmplex_B), cell_count_A(static_cast<int>(coupling_map.size())) {

  const std::size_t num_local_cells_A = coupling_map.size();

  // Need to find the ranks which own the cells in the B mesh for each cell we
  // were passed in the A mesh.

  MPI_Comm B_comm = MPI_COMM_NULL;
  PETSCCHK(PetscObjectGetComm((PetscObject)dmplex_B, &B_comm));

  int rank = 0;
  MPICHK(MPI_Comm_rank(B_comm, &rank));

  auto [global_point_min, cell_owners] =
      get_map_from_global_cell_points_to_ranks(dmplex_B);

  // For each remote rank create an edge in the neighbour graph.

  std::map<int, std::vector<int>> map_remote_rank_to_cells;
  std::set<int> remote_ranks_set;

  int cell_index_A = 0;
  for (const auto &cell_coupling_map : coupling_map) {
    for (const auto &map_entry : cell_coupling_map) {
      const int global_point_index = map_entry.cell_index;

      const int index = global_point_index - global_point_min;
      const int remote_rank = cell_owners.at(index);
      remote_ranks_set.insert(remote_rank);
      map_remote_rank_to_cells[remote_rank].push_back(global_point_index);
    }
    cell_index_A++;
  }

  std::vector<int> remote_ranks_vector;
  remote_ranks_vector.reserve(remote_ranks_set.size());
  std::for_each(remote_ranks_set.begin(), remote_ranks_set.end(),
                [&](auto ix) { remote_ranks_vector.push_back(ix); });

  SuppressMPINullPtrCheck snpc;

  const int degrees = static_cast<int>(remote_ranks_set.size());
  MPICHK(MPI_Dist_graph_create(B_comm, 1, &rank, &degrees,
                               snpc.get(remote_ranks_vector), MPI_UNWEIGHTED,
                               MPI_INFO_NULL, 0, &this->comm_forward));

  // Reverse the neighbour graph direction for the backwards transfer.
  MPICHK(reverse_graph_edge_directions(B_comm, this->comm_forward,
                                       &this->comm_backward));

  // Exchange the cell counts for each edge (can use the forward graph)

  // Exchange the cell indices for each edge
}

void DMPlexMeshCouplerDG0::forward_transfer(std::vector<REAL> &dofs_A,
                                            std::vector<REAL> &dofs_B) {}

void DMPlexMeshCouplerDG0::backward_transfer(std::vector<REAL> &dofs_B,
                                             std::vector<REAL> &dofs_A) {}

void DMPlexMeshCouplerDG0::free() {
  if (this->comm_forward != MPI_COMM_NULL) {
    MPICHK(MPI_Comm_free(&this->comm_forward));
    this->comm_forward = MPI_COMM_NULL;
  }
  if (this->comm_backward != MPI_COMM_NULL) {
    MPICHK(MPI_Comm_free(&this->comm_backward));
    this->comm_backward = MPI_COMM_NULL;
  }
}

} // namespace NESO::Particles::PetscInterface

#endif
