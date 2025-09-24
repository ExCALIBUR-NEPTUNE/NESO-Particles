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

} // namespace NESO::Particles
