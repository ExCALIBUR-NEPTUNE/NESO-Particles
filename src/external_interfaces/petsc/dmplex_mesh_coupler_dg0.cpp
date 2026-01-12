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

  std::map<int, std::vector<int>> map_remote_rank_to_cells_A;
  std::map<int, std::vector<int>> map_remote_rank_to_cells_B;
  std::map<int, std::vector<REAL>> map_remote_rank_to_forward_weights;
  std::map<int, std::vector<REAL>> map_remote_rank_to_backward_weights;
  std::set<int> remote_ranks_set;

  int cell_index_A = 0;
  for (const auto &cell_coupling_map : coupling_map) {
    for (const auto &map_entry : cell_coupling_map) {
      const int global_point_index = map_entry.cell_index;

      const int index = global_point_index - global_point_min;
      const int remote_rank = cell_owners.at(index);
      remote_ranks_set.insert(remote_rank);
      map_remote_rank_to_cells_A[remote_rank].push_back(cell_index_A);
      map_remote_rank_to_cells_B[remote_rank].push_back(global_point_index);
      map_remote_rank_to_forward_weights[remote_rank].push_back(
          map_entry.weight_forward);
      map_remote_rank_to_backward_weights[remote_rank].push_back(
          map_entry.weight_backward);
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

  // The MPI implementation may have re-ordered our edges, hence we retrieve
  // them again.

  int indegree = 0;
  int outdegree = 0;
  int weighted = 0;
  MPICHK(MPI_Dist_graph_neighbors_count(this->comm_forward, &indegree,
                                        &outdegree, &weighted));
  this->sources_forward.resize(indegree);
  this->destinations_forward.resize(outdegree);

  std::vector<int> source_weights(indegree);
  std::vector<int> destination_weights(outdegree);

  MPICHK(MPI_Dist_graph_neighbors(
      this->comm_forward, indegree, snpc.get(this->sources_forward),
      snpc.get(source_weights), outdegree, snpc.get(this->destinations_forward),
      snpc.get(destination_weights)));

  MPICHK(MPI_Dist_graph_neighbors_count(this->comm_backward, &indegree,
                                        &outdegree, &weighted));
  this->sources_backward.resize(indegree);
  this->destinations_backward.resize(outdegree);

  source_weights.resize(indegree);
  destination_weights.resize(outdegree);

  MPICHK(MPI_Dist_graph_neighbors(
      this->comm_backward, indegree, snpc.get(this->sources_backward),
      snpc.get(source_weights), outdegree,
      snpc.get(this->destinations_backward), snpc.get(destination_weights)));

  // Exchange the cell counts for each edge (can use the forward graph)
  this->send_counts_forward.resize(this->destinations_forward.size());
  this->recv_counts_forward.resize(this->sources_forward.size());

  int index = 0;
  for (auto rankx : this->destinations_forward) {
    send_counts_forward.at(index) =
        static_cast<int>(map_remote_rank_to_cells_B.at(rankx).size());
    index++;
  }

  MPICHK(NP_MPI_Neighbor_alltoall_wrapper(
      snpc.get(send_counts_forward), 1, MPI_INT, snpc.get(recv_counts_forward),
      1, MPI_INT, this->comm_forward));

  this->send_disps_forward.resize(this->destinations_forward.size());
  this->recv_disps_forward.resize(this->sources_forward.size());

  std::exclusive_scan(send_counts_forward.begin(), send_counts_forward.end(),
                      this->send_disps_forward.begin(), 0);
  std::exclusive_scan(recv_counts_forward.begin(), recv_counts_forward.end(),
                      this->recv_disps_forward.begin(), 0);

  std::vector<MPI_Aint> send_disps_forward_int(this->send_disps_forward.size());
  std::vector<MPI_Aint> recv_disps_forward_int(this->recv_disps_forward.size());

  std::transform(this->send_disps_forward.begin(),
                 this->send_disps_forward.end(), send_disps_forward_int.begin(),
                 [](auto ix) { return ix * sizeof(int); });

  std::transform(this->recv_disps_forward.begin(),
                 this->recv_disps_forward.end(), recv_disps_forward_int.begin(),
                 [](auto ix) { return ix * sizeof(int); });

  this->send_disps_forward_real.resize(this->send_disps_forward.size());
  this->recv_disps_forward_real.resize(this->recv_disps_forward.size());

  std::transform(this->send_disps_forward.begin(),
                 this->send_disps_forward.end(),
                 this->send_disps_forward_real.begin(),
                 [](auto ix) { return ix * sizeof(REAL); });

  std::transform(this->recv_disps_forward.begin(),
                 this->recv_disps_forward.end(),
                 this->recv_disps_forward_real.begin(),
                 [](auto ix) { return ix * sizeof(REAL); });

  this->total_num_send_cells_forward = std::accumulate(
      send_counts_forward.begin(), send_counts_forward.end(), 0);
  this->total_num_recv_cells_forward = std::accumulate(
      recv_counts_forward.begin(), recv_counts_forward.end(), 0);

  // These are the remote cells of B that this rank will send contributions to.
  std::vector<int> remote_cells;
  remote_cells.reserve(total_num_send_cells_forward);
  this->cells_forward_A.reserve(total_num_send_cells_forward);
  this->weights_forward_A.reserve(total_num_send_cells_forward);

  for (auto rankx : this->destinations_forward) {
    auto &a = map_remote_rank_to_cells_A.at(rankx);
    this->cells_forward_A.insert(this->cells_forward_A.end(), a.begin(),
                                 a.end());
    auto &b = map_remote_rank_to_cells_B.at(rankx);
    remote_cells.insert(remote_cells.end(), b.begin(), b.end());

    auto &wf = map_remote_rank_to_forward_weights.at(rankx);
    this->weights_forward_A.insert(this->weights_forward_A.end(), wf.begin(),
                                   wf.end());
  }

  // These are the local B cell indicies which remote ranks will send
  // contributions to in the order contributions are received.
  this->cells_forward_B.resize(total_num_recv_cells_forward);

  std::vector<MPI_Datatype> send_types(this->destinations_forward.size());
  std::fill(send_types.begin(), send_types.end(), MPI_INT);

  std::vector<MPI_Datatype> recv_types(this->sources_forward.size());
  std::fill(recv_types.begin(), recv_types.end(), MPI_INT);

  // Exchange the cell indices for each edge
  MPICHK(NP_MPI_Neighbor_alltoallw_wrapper(
      snpc.get(remote_cells), snpc.get(send_counts_forward),
      snpc.get(send_disps_forward_int), snpc.get(send_types),
      snpc.get(this->cells_forward_B), snpc.get(recv_counts_forward),
      snpc.get(recv_disps_forward_int), snpc.get(recv_types),
      this->comm_forward));

  // Temporaries to be able to invert the maps from the forward direction to
  // reverse direction.
  std::map<int, int> map_rank_to_forward_index_source;
  std::map<int, int> map_rank_to_forward_index_dest;
  index = 0;
  for (const int rankx : this->sources_forward) {
    map_rank_to_forward_index_source[rankx] = index;
    index++;
  }
  index = 0;
  for (const int rankx : this->destinations_forward) {
    map_rank_to_forward_index_dest[rankx] = index;
    index++;
  }

  this->cells_backward_A.reserve(this->cells_forward_A.size());
  this->weights_backward_A.reserve(this->weights_forward_A.size());

  this->send_disps_backward.reserve(this->destinations_backward.size());
  this->recv_disps_backward.reserve(this->sources_backward.size());

  this->send_dtypes.resize(std::max(this->destinations_forward.size(),
                                    this->destinations_backward.size()));
  this->recv_dtypes.resize(
      std::max(this->sources_forward.size(), this->sources_backward.size()));

  auto lambda_fill_dtypes_real = [](auto &vx) {
    std::fill(vx.begin(), vx.end(), map_ctype_mpi_type<REAL>());
  };
  lambda_fill_dtypes_real(this->send_dtypes);
  lambda_fill_dtypes_real(this->recv_dtypes);

  this->send_counts_backward.reserve(this->destinations_backward.size());
  this->recv_counts_forward.reserve(this->sources_backward.size());

  int offset = 0;
  for (const int rankx : this->sources_backward) {
    const int forward_edge_index = map_rank_to_forward_index_dest.at(rankx);
    const int num_cells = send_counts_forward.at(forward_edge_index);
    const int forward_disp =
        static_cast<int>(this->send_disps_forward.at(forward_edge_index));

    this->cells_backward_A.insert(this->cells_backward_A.end(),
                                  this->cells_forward_A.begin() + forward_disp,
                                  this->cells_forward_A.begin() + forward_disp +
                                      num_cells);

    auto &wb = map_remote_rank_to_backward_weights.at(rankx);
    NESOASSERT(static_cast<int>(wb.size()) == num_cells,
               "Failure to map to correct weights.");
    this->weights_backward_A.insert(this->weights_backward_A.end(), wb.begin(),
                                    wb.end());
    this->recv_disps_backward.push_back(offset);
    this->recv_counts_backward.push_back(num_cells);
    offset += num_cells;
  }
  this->total_num_recv_cells_backward = offset;

  this->cells_backward_B.reserve(this->cells_forward_B.size());
  offset = 0;
  for (const int rankx : this->destinations_backward) {
    const int forward_edge_index = map_rank_to_forward_index_source.at(rankx);
    const int num_cells = recv_counts_forward.at(forward_edge_index);

    const int forward_disp =
        static_cast<int>(this->recv_disps_forward.at(forward_edge_index));

    this->cells_backward_B.insert(this->cells_backward_B.end(),
                                  this->cells_forward_B.begin() + forward_disp,
                                  this->cells_forward_B.begin() + forward_disp +
                                      num_cells);
    this->send_disps_backward.push_back(offset);
    this->send_counts_backward.push_back(num_cells);
    offset += num_cells;
  }
  this->total_num_send_cells_backward = offset;

  this->send_disps_backward_real.resize(this->send_disps_backward.size());
  this->recv_disps_backward_real.resize(this->recv_disps_backward.size());

  std::transform(this->send_disps_backward.begin(),
                 this->send_disps_backward.end(),
                 this->send_disps_backward_real.begin(),
                 [](auto ix) { return ix * sizeof(REAL); });

  std::transform(this->recv_disps_backward.begin(),
                 this->recv_disps_backward.end(),
                 this->recv_disps_backward_real.begin(),
                 [](auto ix) { return ix * sizeof(REAL); });

  // Convert the b cell indices back to local indices.
  DMPlexHelper dmh_B(B_comm, dmplex_B);

  this->cell_count_B = dmh_B.get_cell_count();
  auto lambda_global_to_local = [&](std::vector<int> &cells) {
    std::for_each(cells.begin(), cells.end(), [&](auto &cellx) {
      cellx = dmh_B.get_local_point_from_global_point(cellx);
    });
  };

  lambda_global_to_local(this->cells_forward_B);
  lambda_global_to_local(this->cells_backward_B);

  NESOASSERT(this->cells_forward_A.size() == this->weights_forward_A.size(),
             "Vector size missmatch forward.");
  NESOASSERT(this->cells_backward_A.size() == this->weights_backward_A.size(),
             "Vector size missmatch forward.");

  this->h_stage_send.resize(std::max(this->total_num_send_cells_forward,
                                     this->total_num_send_cells_backward));
  this->h_stage_recv.resize(std::max(this->total_num_recv_cells_forward,
                                     this->total_num_recv_cells_backward));
}

void DMPlexMeshCouplerDG0::forward_transfer(std::vector<REAL> &dofs_A,
                                            std::vector<REAL> &dofs_B) {

  NESOASSERT(dofs_A.size() >= this->cell_count_A,
             "Insufficent number of DOFs passed for A.");
  NESOASSERT(dofs_B.size() >= this->cell_count_B,
             "Insufficent number of DOFs passed for B.");

  const std::size_t NA = this->cells_forward_A.size();
  for (int ix = 0; ix < NA; ix++) {
    const int index_A = this->cells_forward_A[ix];
    const REAL weight_A = this->weights_forward_A[ix];
    this->h_stage_send[ix] = dofs_A[index_A] * weight_A;
  }

  SuppressMPINullPtrCheck snpc;
  MPICHK(NP_MPI_Neighbor_alltoallw_wrapper(
      snpc.get(this->h_stage_send), snpc.get(this->send_counts_forward),
      snpc.get(this->send_disps_forward_real), snpc.get(this->send_dtypes),
      snpc.get(this->h_stage_recv), snpc.get(this->recv_counts_forward),
      snpc.get(this->recv_disps_forward_real), snpc.get(this->recv_dtypes),
      this->comm_forward));

  const int NB = static_cast<int>(this->cells_forward_B.size());
  std::fill(dofs_B.begin(), dofs_B.end(), 0.0);
  for (int ix = 0; ix < NB; ix++) {
    const int index_B = this->cells_forward_B[ix];
    dofs_B[index_B] += this->h_stage_recv[ix];
  }
}

void DMPlexMeshCouplerDG0::backward_transfer(std::vector<REAL> &dofs_B,
                                             std::vector<REAL> &dofs_A) {

  NESOASSERT(dofs_A.size() >= this->cell_count_A,
             "Insufficent number of DOFs passed for A.");
  NESOASSERT(dofs_B.size() >= this->cell_count_B,
             "Insufficent number of DOFs passed for B.");

  const int NB = static_cast<int>(this->cells_backward_B.size());
  for (int ix = 0; ix < NB; ix++) {
    const int index_B = this->cells_backward_B[ix];
    this->h_stage_send[ix] = dofs_B[index_B];
  }

  SuppressMPINullPtrCheck snpc;
  MPICHK(NP_MPI_Neighbor_alltoallw_wrapper(
      snpc.get(this->h_stage_send), snpc.get(this->send_counts_backward),
      snpc.get(this->send_disps_backward_real), snpc.get(this->send_dtypes),
      snpc.get(this->h_stage_recv), snpc.get(this->recv_counts_backward),
      snpc.get(this->recv_disps_backward_real), snpc.get(this->recv_dtypes),
      this->comm_backward));

  const int NA = static_cast<int>(this->cells_backward_A.size());
  std::fill(dofs_A.begin(), dofs_A.end(), 0.0);
  for (int ix = 0; ix < NA; ix++) {
    const int index_A = this->cells_backward_A[ix];
    const REAL weight_A = this->weights_backward_A[ix];
    dofs_A[index_A] += this->h_stage_recv[ix] * weight_A;
  }
}

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
