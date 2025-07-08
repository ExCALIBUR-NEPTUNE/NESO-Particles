#ifndef _NESO_PARTICLES_BOUNDARY_BOUNDARY_MESH_INTERFACE_HPP_
#define _NESO_PARTICLES_BOUNDARY_BOUNDARY_MESH_INTERFACE_HPP_

#include "../communication.hpp"
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace NESO::Particles {

class BoundaryMeshInterface {
protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  struct {
    // Original MPI Communicator to use.
    MPI_Comm comm{MPI_COMM_NULL};
    // Neighbour comm for boundary exchanges.
    MPI_Comm ncomm{MPI_COMM_NULL};
    // Map from remote MPI ranks to geometry ids that those remote ranks own but
    // this rank holds a contribution for.
    std::map<int, std::set<int>> map_recv_rank_to_geom_ids;
    // Map from remote ranks which hold copies of geometry ids this rank owns
    // and the geometry objects that they own.
    std::map<int, std::set<int>> map_send_rank_to_geom_ids;
    // Number of geometery ids each remote rank will send to this rank. Ordering
    // is defined by the MPI graph.
    std::vector<int> incoming_geom_counts;

    struct {
      int indegree;
      int outdegree;
      std::vector<int> sources;
      std::vector<int> destinations;
    } graph;

  } boundary;

  void boundary_init(MPI_Comm comm);
  void boundary_free();

public:
  /**
   * Indicate to the implementation that communication patterns should be set up
   * with the owning ranks of geometry objects. This function must be called
   * collectively on the mesh communicator.
   *
   * @param rank_geom_ids Vector of {<rank>, <geometry id>} pairs which this
   * rank has collected and may push data back to the original rank.
   */
  void boundary_extend_exchange_pattern(
      const std::vector<std::pair<int, int>> &rank_geom_ids);
};

} // namespace NESO::Particles

#endif
