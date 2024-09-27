#ifndef _NESO_PARTICLES_COMMUNICATION_UTILITY_HPP
#define _NESO_PARTICLES_COMMUNICATION_UTILITY_HPP

#include "communication_typedefs.hpp"
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
inline void gather_v(std::vector<T> &input_vector, MPI_Comm comm,
                     const int root, std::vector<T> &output_vector) {

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

/**
 * MPI_Allgatherv for std::vector. Collective on the communicator.
 *
 * @param[in] input_vector Vector to send to other ranks.
 * @param[in] comm MPI communicator to use.
 * @param[in, out] output_vector Output vector to gather input vectors on.
 */
template <typename T>
inline void all_gather_v(std::vector<T> &input_vector, MPI_Comm comm,
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

} // namespace NESO::Particles

#endif
