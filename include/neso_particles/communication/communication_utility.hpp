#ifndef _NESO_PARTICLES_COMMUNICATION_UTILITY_HPP
#define _NESO_PARTICLES_COMMUNICATION_UTILITY_HPP

#include "communication_typedefs.hpp"
#include <set>
#include <vector>

namespace NESO::Particles {

/**
 * MPI_Gatherv for std::vector. Collective on the communicator.
 *
 * @param[in] input_vector Vector to send to root rank.
 * @param[in] comm MPI communicator to use.
 * @param[in] root Root MPI rank to gather input vectors on.
 * @param[in, out] output_vector Output vector to gather input vectors on.
 */
template <typename T>
void gather_v(std::vector<T> &input_vector, MPI_Comm comm, const int root,
              std::vector<T> &output_vector) {

  int size, rank;
  MPICHK(MPI_Comm_size(comm, &size));
  MPICHK(MPI_Comm_rank(comm, &rank));

  const bool root_rank = rank == root;
  const int masked_size = root_rank ? size : 0;
  std::vector<int> recv_counts(masked_size);
  std::vector<int> displacements(masked_size);
  std::fill(recv_counts.begin(), recv_counts.end(), 0);
  std::iota(displacements.begin(), displacements.end(), 0);

  const int input_count = input_vector.size();

  MPICHK(MPI_Gather(&input_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                    root, comm));

  const int output_count =
      root_rank ? std::reduce(recv_counts.begin(), recv_counts.end()) : 0;
  output_vector.resize(output_count);

  std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                      displacements.begin(), 0);

  MPICHK(MPI_Gatherv(input_vector.data(), input_count, map_ctype_mpi_type<T>(),
                     output_vector.data(), recv_counts.data(),
                     displacements.data(), map_ctype_mpi_type<T>(), root,
                     comm));
}

extern template void gather_v(std::vector<REAL> &input_vector, MPI_Comm comm,
                              const int root, std::vector<REAL> &output_vector);
extern template void gather_v(std::vector<INT> &input_vector, MPI_Comm comm,
                              const int root, std::vector<INT> &output_vector);
extern template void gather_v(std::vector<int> &input_vector, MPI_Comm comm,
                              const int root, std::vector<int> &output_vector);

/**
 * MPI_Allgatherv for std::vector. Collective on the communicator.
 *
 * @param[in] input_vector Vector to send to other ranks.
 * @param[in] comm MPI communicator to use.
 * @param[in, out] output_vector Output vector to gather input vectors on.
 */
template <typename T>
void all_gather_v(std::vector<T> &input_vector, MPI_Comm comm,
                  std::vector<T> &output_vector) {

  int size, rank;
  MPICHK(MPI_Comm_size(comm, &size));
  MPICHK(MPI_Comm_rank(comm, &rank));

  std::vector<int> recv_counts(size);
  std::vector<int> displacements(size);
  std::fill(recv_counts.begin(), recv_counts.end(), 0);
  std::iota(displacements.begin(), displacements.end(), 0);

  const int input_count = input_vector.size();
  MPICHK(MPI_Allgather(&input_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                       comm));

  const int output_count = std::reduce(recv_counts.begin(), recv_counts.end());
  output_vector.resize(output_count);
  std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                      displacements.begin(), 0);

  MPICHK(MPI_Allgatherv(input_vector.data(), input_count,
                        map_ctype_mpi_type<T>(), output_vector.data(),
                        recv_counts.data(), displacements.data(),
                        map_ctype_mpi_type<T>(), comm));
}

extern template void all_gather_v(std::vector<REAL> &input_vector,
                                  MPI_Comm comm,
                                  std::vector<REAL> &output_vector);
extern template void all_gather_v(std::vector<INT> &input_vector, MPI_Comm comm,
                                  std::vector<INT> &output_vector);
extern template void all_gather_v(std::vector<int> &input_vector, MPI_Comm comm,
                                  std::vector<int> &output_vector);

/**
 * Create a new graph communicator identical to an existing graph communicator
 * except the edge directions are reversed. Collective on the communicators.
 *
 * @param[in] comm Original communicator graph_comm is created from.
 * @param[in] graph_comm Graph communicator to duplicate with edge direction
 * swapped.
 * @param[in, out] reverse_graph_comm New graph communicator with the same edges
 * as comm but with the directions reversed.
 * @returns MPI_SUCCESS on success.
 */
[[nodiscard]] int reverse_graph_edge_directions(MPI_Comm comm,
                                                MPI_Comm graph_comm,
                                                MPI_Comm *reverse_graph_comm);

/**
 * Send the contents of a std::set to another MPI rank.
 *
 * @param send_set Set to send.
 * @param dest Receiving MPI rank that must call set_recv.
 * @param tag Tag for MPI send operation.
 * @param comm MPI Communicator.
 */
template <typename T>
void set_send(std::set<T> &send_set, const int dest, const int tag,
              MPI_Comm comm) {
  const int send_count = static_cast<int>(send_set.size());
  MPICHK(MPI_Send(&send_count, 1, MPI_INT, dest, tag, comm));

  if (send_count) {
    std::vector<T> send_vector;
    send_vector.reserve(send_count);
    for (auto ix : send_set) {
      send_vector.push_back(ix);
    }

    MPICHK(MPI_Send(send_vector.data(), send_count, map_ctype_mpi_type<T>(),
                    dest, tag, comm));
  }
}

/**
 * Receive the contents of a std::set from another MPI rank.
 *
 * @param source Sending MPI rank that must call set_send.
 * @param tag Tag for MPI send operation.
 * @param comm MPI Communicator.
 * @returns Set recieved from remote MPI rank.
 */
template <typename T>
std::set<T> set_recv(const int source, const int tag, MPI_Comm comm) {
  int recv_count = 0;
  MPICHK(
      MPI_Recv(&recv_count, 1, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE));

  std::set<T> return_set{};
  if (recv_count) {
    std::vector<T> recv_vector(recv_count);
    MPICHK(MPI_Recv(recv_vector.data(), recv_count, map_ctype_mpi_type<T>(),
                    source, tag, comm, MPI_STATUS_IGNORE));
    for (int ix : recv_vector) {
      return_set.insert(ix);
    }
  }

  return return_set;
}

extern template std::set<int> set_recv(const int source, const int tag,
                                       MPI_Comm comm);

/**
 * Broadcast the contents of a std::set to all other MPI ranks. Collective on
 * the communicator.
 *
 * @param bcast_set Set to broadcast.
 * @param root Root rank for broadcast operation.
 * @param comm MPI Communicator.
 * @returns Broadcast set from remote root rank.
 */
template <typename T>
std::set<T> set_bcast(std::set<T> &bcast_set, int root, MPI_Comm comm) {
  int set_size = static_cast<int>(bcast_set.size());
  MPICHK(MPI_Bcast(&set_size, 1, MPI_INT, root, comm));
  std::vector<T> recv_buffer;
  int rank = 0;
  MPICHK(MPI_Comm_rank(comm, &rank));
  if (root == rank) {
    recv_buffer.reserve(set_size);
    for (auto ix : bcast_set) {
      recv_buffer.push_back(ix);
    }
  } else {
    recv_buffer.resize(set_size);
  }

  MPICHK(MPI_Bcast(recv_buffer.data(), set_size, map_ctype_mpi_type<T>(), root,
                   comm));

  std::set<T> return_set;
  for (auto ix : recv_buffer) {
    return_set.insert(ix);
  }

  return return_set;
}

extern template std::set<int> set_bcast(std::set<int> &bcast_set, int root,
                                        MPI_Comm comm);

/**
 * Reduce sets across all MPI ranks using the union operation. Must be called
 * collectively on the communicator.
 *
 * @param set_input The contributing set from this MPI rank.
 * @param root The root MPI rank which will hold the union of all MPI ranks on
 * return.
 * @param comm MPI Communicator.
 * @returns Union of all contributions on the root rank otherwise the empty set.
 */
template <typename T>
std::set<T> set_reduce_union(std::set<T> &set_input, int root, MPI_Comm comm) {

  int rank = 0;
  int size = 0;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  std::set<T> working_set;
  for (auto ix : set_input) {
    working_set.insert(ix);
  }
  if (size == 1) {
    return working_set;
  }

  int reduction_size = 0;
  if (is_power_of_two(size)) {
    reduction_size = size;
  } else {
    reduction_size = get_prev_power_of_two(size);
    int num_remainder_ranks = size - reduction_size;
    if (rank < num_remainder_ranks) {
      const int sending_rank = reduction_size + rank;
      auto tmp_set = set_recv<T>(sending_rank, 0, comm);
      for (auto ix : tmp_set) {
        working_set.insert(ix);
      }
    } else if (rank >= reduction_size) {
      const int recving_rank = rank - reduction_size;
      set_send(working_set, recving_rank, 0, comm);
    }
  }

  for (int s = reduction_size / 2; s > 0; s >>= 1) {
    if (rank < s) {
      const int sending_rank = rank + s;
      auto tmp_set = set_recv<T>(sending_rank, 0, comm);
      for (auto ix : tmp_set) {
        working_set.insert(ix);
      }
    } else if ((rank >= s) && (rank < 2 * s)) {
      const int recving_rank = rank - s;
      set_send(working_set, recving_rank, 0, comm);
    }
  }

  // There should be a way to shift the tree reduce to avoid this send.
  if (root != 0) {
    if (rank == 0) {
      set_send(working_set, root, 0, comm);
    } else if (rank == root) {
      working_set = set_recv<T>(0, 0, comm);
    }
  }

  if (rank == root) {
    return working_set;
  } else {
    return std::set<T>{};
  }
}

extern template std::set<int> set_reduce_union(std::set<int> &set_input,
                                               int root, MPI_Comm comm);

/**
 * Reduce sets across all MPI ranks using the union operation. Must be called
 * collectively on the communicator.
 *
 * @param set_input The contributing set from this MPI rank.
 * @param comm MPI Communicator.
 * @returns Union of all contributions on all ranks.
 */
template <typename T>
std::set<T> set_all_reduce_union(std::set<T> &set_input, MPI_Comm comm) {
  auto tmp = set_reduce_union(set_input, 0, comm);
  return set_bcast(tmp, 0, comm);
}

extern template std::set<int> set_all_reduce_union(std::set<int> &set_input,
                                                   MPI_Comm comm);

/**
 * Reimplementation of MPI_Neighbor_alltoall for broken MPI implementations.
 * OpenMPI seems to be broken when comm is a dist graph created from a cart
 * comm.
 */
int NP_MPI_Neighbor_alltoall(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             MPI_Comm comm);

/**
 * Calls the MPI implementation of MPI_Neighbor_alltoall or
 * NP_MPI_Neighbor_alltoall depending on if
 * NESO_PARTICLES_MPI_NEIGHBOUR_ALL_TO_ALL_FIX is defined.
 */
int NP_MPI_Neighbor_alltoall_wrapper(const void *sendbuf, int sendcount,
                                     MPI_Datatype sendtype, void *recvbuf,
                                     int recvcount, MPI_Datatype recvtype,
                                     MPI_Comm comm);

/**
 * Reimplementation of MPI_Neighbor_alltoall for broken MPI implementations.
 * OpenMPI seems to be broken when comm is a dist graph created from a cart
 * comm.
 */
int NP_MPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[],
                              const MPI_Aint sdispls[],
                              const MPI_Datatype sendtypes[], void *recvbuf,
                              const int recvcounts[], const MPI_Aint rdispls[],
                              const MPI_Datatype recvtypes[], MPI_Comm comm);
/**
 * Calls the MPI implementation of MPI_Neighbor_alltoallw or
 * NP_MPI_Neighbor_alltoallw depending on if
 * NESO_PARTICLES_MPI_NEIGHBOUR_ALL_TO_ALL_FIX is defined.
 */
int NP_MPI_Neighbor_alltoallw_wrapper(
    const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[],
    const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
    const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm);

/**
 * MPI implementations check if the arguments they have been passed are nullptr.
 * In some cases these checks happen even if the pointer is never dereferenced.
 * This is a problem when a buffer is not needed and is backed by a std::vector
 * of size zero. Size zero std::vectors may return a nullptr on call to data().
 *
 * This class returns a fictuous non-zero pointer instead of a nullptr to avoid
 * these checks causing the program to stop for no good reason.
 */
struct SuppressMPINullPtrCheck {
  std::uintptr_t source{1};

  template <typename T> T *get(std::vector<T> &vec) {
    if (vec.size()) {
      return vec.data();
    } else {
      return reinterpret_cast<T *>(this->source++);
    }
  }
};

} // namespace NESO::Particles

#endif
