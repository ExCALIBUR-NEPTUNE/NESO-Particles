#ifndef _NESO_PARTICLES_BOUNDARY_BOUNDARY_MESH_INTERFACE_HPP_
#define _NESO_PARTICLES_BOUNDARY_BOUNDARY_MESH_INTERFACE_HPP_

#include "../communication.hpp"
#include <map>
#include <set>
#include <typeindex>
#include <utility>
#include <vector>

namespace NESO::Particles {

class BoundaryMeshInterface {
protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  struct AllToAllWArgs {
    std::vector<int> sendcounts;
    std::vector<MPI_Aint> sdispls;
    std::vector<MPI_Datatype> sendtypes;
    std::vector<int> recvcounts;
    std::vector<MPI_Aint> rdispls;
    std::vector<MPI_Datatype> recvtypes;
  };

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
    // Number of geometery ids to send to each remote rank. Ordering is defined
    // by the MPI graph.
    std::vector<int> outgoing_geom_counts;
    // Total number of geometry objects for which there is incoming data.
    int total_num_incoming_geoms{0};
    // Total number of geometry objects for which there is outgoing data.
    int total_num_outgoing_geoms{0};
    // Geometry ids of the incoming data
    std::vector<int> incoming_geom_ids;
    // Geometry ids of the outgoing data
    std::vector<int> outgoing_geom_ids;

    std::map<std::pair<std::type_index, int>, AllToAllWArgs>
        map_typencomp_alltoallwargs;

    struct {
      int indegree{0};
      int outdegree{0};
      std::vector<int> sources;
      std::vector<int> destinations;
    } graph;

  } boundary;

  void boundary_init(MPI_Comm comm);
  void boundary_free();

  template <typename T>
  const AllToAllWArgs &boundary_get_alltoallw_args(const int ncomp) {
    if (!this->boundary.map_typencomp_alltoallwargs.count({typeid(T), ncomp})) {

      AllToAllWArgs args;
      const auto outdegree = this->boundary.graph.outdegree;
      const auto indegree = this->boundary.graph.indegree;

      args.sendcounts.resize(std::max(1, outdegree));
      args.recvcounts.resize(std::max(1, indegree));
      args.sendtypes.resize(std::max(1, outdegree));
      args.recvtypes.resize(std::max(1, indegree));
      args.sdispls.resize(std::max(1, outdegree));
      args.rdispls.resize(std::max(1, indegree));

      MPI_Aint offset = 0;
      for (int ix = 0; ix < outdegree; ix++) {
        const int geom_count = this->boundary.outgoing_geom_counts[ix];
        args.sendcounts[ix] = geom_count * ncomp;
        args.sdispls[ix] = offset;
        offset += geom_count * ncomp * sizeof(T);
      }
      offset = 0;
      for (int ix = 0; ix < indegree; ix++) {
        const int geom_count = this->boundary.incoming_geom_counts[ix];
        args.recvcounts[ix] = geom_count * ncomp;
        args.rdispls[ix] = offset;
        offset += geom_count * ncomp * sizeof(T);
      }

      std::fill(args.sendtypes.begin(), args.sendtypes.end(),
                map_ctype_mpi_type<T>());
      std::fill(args.recvtypes.begin(), args.recvtypes.end(),
                map_ctype_mpi_type<T>());

      this->boundary.map_typencomp_alltoallwargs[{typeid(T), ncomp}] = args;
    }
    return this->boundary.map_typencomp_alltoallwargs.at({typeid(T), ncomp});
  }

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

  /**
   * Send data from each rank to the owning rank for the data.
   *
   * @param[in] data Data to send to owning ranks. num_geoms x ncomp sized array
   * ordered by component fastest followed by geometry index. Geometry object id
   * and ordering is defined by outgoing_geom_ids_geom_ids.
   * @param[in] ncomp Number of components to send per geometry object.
   * @param[in, out] data_gathered Output data num_geoms x ncomp sized array
   * ordered by component fastest followed by geometry index. Geometry object id
   * and ordering is defined by incoming_geom_ids.
   */
  template <typename T>
  void boundary_exchange_surface(T *data, const int ncomp, T *data_gathered) {
    if (ncomp < 1) {
      return;
    }
    const AllToAllWArgs &args = this->boundary_get_alltoallw_args<T>(ncomp);

    T null_data = 0;
    T null_data_gathered = 0;

    MPICHK(MPI_Neighbor_alltoallw(
        data != nullptr ? data : &null_data, args.sendcounts.data(),
        args.sdispls.data(), args.sendtypes.data(),
        data_gathered != nullptr ? data_gathered : &null_data_gathered,
        args.recvcounts.data(), args.rdispls.data(), args.recvtypes.data(),
        this->boundary.ncomm));
  }
};

extern template void
BoundaryMeshInterface::boundary_exchange_surface(int *data, const int ncomp,
                                                 int *data_gathered);
extern template void
BoundaryMeshInterface::boundary_exchange_surface(INT *data, const int ncomp,
                                                 INT *data_gathered);
extern template void
BoundaryMeshInterface::boundary_exchange_surface(REAL *data, const int ncomp,
                                                 REAL *data_gathered);

} // namespace NESO::Particles

#endif
