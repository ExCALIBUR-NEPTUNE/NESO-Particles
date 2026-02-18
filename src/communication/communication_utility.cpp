#include <map>
#include <neso_particles/communication/communication_utility.hpp>
#include <vector>

namespace NESO::Particles {

template void gather_v(std::vector<REAL> &input_vector, MPI_Comm comm,
                       const int root, std::vector<REAL> &output_vector);
template void gather_v(std::vector<INT> &input_vector, MPI_Comm comm,
                       const int root, std::vector<INT> &output_vector);
template void gather_v(std::vector<int> &input_vector, MPI_Comm comm,
                       const int root, std::vector<int> &output_vector);

template void all_gather_v(std::vector<REAL> &input_vector, MPI_Comm comm,
                           std::vector<REAL> &output_vector);
template void all_gather_v(std::vector<INT> &input_vector, MPI_Comm comm,
                           std::vector<INT> &output_vector);
template void all_gather_v(std::vector<int> &input_vector, MPI_Comm comm,
                           std::vector<int> &output_vector);

int reverse_graph_edge_directions(MPI_Comm comm, MPI_Comm graph_comm,
                                  MPI_Comm *reverse_graph_comm) {
  int indegree = -1, outdegree = -1, weighted = -1;

  MPICHK(MPI_Dist_graph_neighbors_count(graph_comm, &indegree, &outdegree,
                                        &weighted));
  std::vector<int> sources(indegree);
  std::vector<int> sourcesweights(indegree);
  std::vector<int> destinations(outdegree);
  std::vector<int> destweights(outdegree);

  MPICHK(MPI_Dist_graph_neighbors(graph_comm, indegree, sources.data(),
                                  sourcesweights.data(), outdegree,
                                  destinations.data(), destweights.data()));

  // We swap out and in for this function call to reverse the graph.
  return MPI_Dist_graph_create_adjacent(comm, outdegree, destinations.data(),
                                        destweights.data(), indegree,
                                        sources.data(), sourcesweights.data(),
                                        MPI_INFO_NULL, 0, reverse_graph_comm);
}

template std::set<int> set_recv(const int source, const int tag, MPI_Comm comm);
template std::set<int> set_bcast(std::set<int> &bcast_set, int root,
                                 MPI_Comm comm);
template std::set<int> set_reduce_union(std::set<int> &set_input, int root,
                                        MPI_Comm comm);
template std::set<int> set_all_reduce_union(std::set<int> &set_input,
                                            MPI_Comm comm);

int NP_MPI_Neighbor_alltoall(const void *sendbuf, int sendcount,
                             MPI_Datatype sendtype, void *recvbuf,
                             int recvcount, MPI_Datatype recvtype,
                             MPI_Comm comm) {

  int indegree = -1, outdegree = -1, weighted = -1;
  MPICHK(
      MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted));
  std::vector<int> sources(std::max(indegree, 1));
  std::vector<int> sourcesweights(std::max(indegree, 1));
  std::vector<int> destinations(std::max(outdegree, 1));
  std::vector<int> destweights(std::max(outdegree, 1));
  MPICHK(MPI_Dist_graph_neighbors(comm, indegree, sources.data(),
                                  sourcesweights.data(), outdegree,
                                  destinations.data(), destweights.data()));
  std::vector<MPI_Request> recv_requests(std::max(indegree, 1));
  std::vector<MPI_Request> send_requests(std::max(outdegree, 1));

  {
    int recv_bytes = -1;
    MPICHK(MPI_Type_size(recvtype, &recv_bytes));
    std::map<int, int> recv_tags;
    for (int ix = 0; ix < indegree; ix++) {
      const int source = sources[ix];
      const int tag = recv_tags[source];
      recv_tags[source] = recv_tags[source] + 1;

      MPICHK(MPI_Irecv(
          static_cast<char *>(recvbuf) + recvcount * recv_bytes * ix, recvcount,
          recvtype, source, tag, comm, &recv_requests[ix]));
    }
  }

  {
    int send_bytes = -1;
    MPICHK(MPI_Type_size(sendtype, &send_bytes));
    std::map<int, int> send_tags;

    for (int ix = 0; ix < outdegree; ix++) {
      const int dest = destinations[ix];
      const int tag = send_tags[dest];
      send_tags[dest] = send_tags[dest] + 1;

      MPICHK(MPI_Isend(
          static_cast<const char *>(sendbuf) + sendcount * send_bytes * ix,
          sendcount, sendtype, dest, tag, comm, &send_requests[ix]));
    }
  }

  MPICHK(MPI_Waitall(outdegree, send_requests.data(), MPI_STATUSES_IGNORE));
  MPICHK(MPI_Waitall(indegree, recv_requests.data(), MPI_STATUSES_IGNORE));

  return MPI_SUCCESS;
}

int NP_MPI_Neighbor_alltoall_wrapper(const void *sendbuf, int sendcount,
                                     MPI_Datatype sendtype, void *recvbuf,
                                     int recvcount, MPI_Datatype recvtype,
                                     MPI_Comm comm) {
#ifdef NESO_PARTICLES_MPI_NEIGHBOUR_ALL_TO_ALL_FIX

  return NP_MPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf,
                                  recvcount, recvtype, comm);

#else
  return MPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                               recvtype, comm);
#endif
}

int NP_MPI_Neighbor_alltoallw(const void *sendbuf, const int sendcounts[],
                              const MPI_Aint sdispls[],
                              const MPI_Datatype sendtypes[], void *recvbuf,
                              const int recvcounts[], const MPI_Aint rdispls[],
                              const MPI_Datatype recvtypes[], MPI_Comm comm) {

  int indegree = -1, outdegree = -1, weighted = -1;
  MPICHK(
      MPI_Dist_graph_neighbors_count(comm, &indegree, &outdegree, &weighted));
  std::vector<int> sources(std::max(indegree, 1));
  std::vector<int> sourcesweights(std::max(indegree, 1));
  std::vector<int> destinations(std::max(outdegree, 1));
  std::vector<int> destweights(std::max(outdegree, 1));
  MPICHK(MPI_Dist_graph_neighbors(comm, indegree, sources.data(),
                                  sourcesweights.data(), outdegree,
                                  destinations.data(), destweights.data()));
  std::vector<MPI_Request> recv_requests(std::max(indegree, 1));
  std::vector<MPI_Request> send_requests(std::max(outdegree, 1));

  {
    std::map<int, int> recv_tags;
    for (int ix = 0; ix < indegree; ix++) {
      const int source = sources[ix];
      const int tag = recv_tags[source];
      recv_tags[source] = recv_tags[source] + 1;

      MPICHK(MPI_Irecv(static_cast<char *>(recvbuf) + rdispls[ix],
                       recvcounts[ix], recvtypes[ix], source, tag, comm,
                       &recv_requests[ix]));
    }
  }

  {
    std::map<int, int> send_tags;
    for (int ix = 0; ix < outdegree; ix++) {
      const int dest = destinations[ix];
      const int tag = send_tags[dest];
      send_tags[dest] = send_tags[dest] + 1;

      MPICHK(MPI_Isend(static_cast<const char *>(sendbuf) + sdispls[ix],
                       sendcounts[ix], sendtypes[ix], dest, tag, comm,
                       &send_requests[ix]));
    }
  }

  MPICHK(MPI_Waitall(outdegree, send_requests.data(), MPI_STATUSES_IGNORE));
  MPICHK(MPI_Waitall(indegree, recv_requests.data(), MPI_STATUSES_IGNORE));

  return MPI_SUCCESS;
}

int NP_MPI_Neighbor_alltoallw_wrapper(
    const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[],
    const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
    const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm) {

#ifdef NESO_PARTICLES_MPI_NEIGHBOUR_ALL_TO_ALL_FIX
  return NP_MPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                   recvbuf, recvcounts, rdispls, recvtypes,
                                   comm);

#else
  return MPI_Neighbor_alltoallw(sendbuf, sendcounts, sdispls, sendtypes,
                                recvbuf, recvcounts, rdispls, recvtypes, comm);
#endif
}

std::vector<int> get_cart_dims(const int size, const int ndim) {
  std::vector<int> mpi_dims(ndim);
  std::fill(mpi_dims.begin(), mpi_dims.end(), 0);
  if (ndim > 0) {
    MPICHK(MPI_Dims_create(size, ndim, mpi_dims.data()));
  }
  return mpi_dims;
}

std::vector<int> get_reordered_cart_decomp(const int ndim,
                                           std::vector<int> mpi_dims,
                                           std::vector<int> &cell_counts) {

  auto cell_count_ordering = reverse_argsort(cell_counts);
  std::sort(mpi_dims.begin(), mpi_dims.end(), std::greater<int>());

  std::vector<int> mpi_dims_reordered(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    mpi_dims_reordered.at(cell_count_ordering.at(dimx)) = mpi_dims.at(dimx);
  }

  return mpi_dims_reordered;
}

std::vector<int>
get_lower_dimension_cart_decomp(const int size, const int ndim,
                                std::vector<int> &cell_counts) {

  int effective_ndim = ndim;
  int effective_size = size;

  std::vector<int> known_dims(ndim);
  std::fill(known_dims.begin(), known_dims.end(), -1);

  std::vector<int> reduced_space;
  reduced_space.reserve(ndim);

  for (int dimx = 0; dimx < ndim; dimx++) {
    const int cell_count = cell_counts[dimx];
    if (effective_size % cell_count == 0) {
      effective_ndim--;
      effective_size /= cell_count;
      known_dims.at(dimx) = cell_count;
    } else {
      reduced_space.push_back(cell_count);
    }
  }

  // If the decomp factors the problem in all dimensions already then we are
  // done.
  if (effective_ndim == 0) {
    return known_dims;
  }

  if (effective_ndim != ndim) {
    auto tmp = get_reordered_cart_decomp(
        effective_ndim, get_cart_dims(effective_size, effective_ndim),
        reduced_space);

    int index = 0;
    for (int dimx = 0; dimx < ndim; dimx++) {
      if (known_dims.at(dimx) < 1) {
        known_dims.at(dimx) = tmp[index];
        index++;
      }
    }

    return known_dims;
  }

  return get_reordered_cart_decomp(ndim, get_cart_dims(size, ndim),
                                   cell_counts);
}

} // namespace NESO::Particles
