#ifndef _NESO_PARTICLES_BOUNDARY_BOUNDARY_MESH_INTERFACE_HPP_
#define _NESO_PARTICLES_BOUNDARY_BOUNDARY_MESH_INTERFACE_HPP_

#include "../communication.hpp"
#include "../compute_target.hpp"
#include "../containers/blocked_binary_tree.hpp"
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
    // Compute device.
    SYCLTargetSharedPtr sycl_target;
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
    // Counter to place an ordering on face geom ids;
    INT geom_counter{0};
    // Map from linear sequential index to geom id.
    std::map<INT, INT> map_linear_index_to_geom_id;
    // Map from geom id to linear sequential index.
    std::map<INT, INT> map_geom_id_to_linear_index;
    // Device map from geom id to linear sequential index.
    std::shared_ptr<BlockedBinaryTree<INT, INT>> d_map_geom_id_to_linear_index;

    std::map<std::pair<std::type_index, int>, AllToAllWArgs>
        map_typencomp_alltoallwargs;

    struct {
      int indegree{0};
      int outdegree{0};
      std::vector<int> sources;
      std::vector<int> destinations;
    } graph;

  } boundary;

  template <typename T>
  const AllToAllWArgs &get_alltoallw_args(const int ncomp) {
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
  BoundaryMeshInterface() = default;

  /**
   * @param comm MPI communicator for mesh.
   * @param sycl_target Compute device for device maps.
   */
  BoundaryMeshInterface(MPI_Comm comm, SYCLTargetSharedPtr sycl_target);

  /**
   * Indicate to the implementation that communication patterns should be set up
   * with the owning ranks of geometry objects. This function must be called
   * collectively on the mesh communicator.
   *
   * @param rank_geom_ids Vector of {<rank>, <geometry id>} pairs which this
   * rank has collected and may push data back to the original rank.
   */
  void extend_exchange_pattern(
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
  void exchange_surface(T *data, const int ncomp, T *data_gathered) {
    if (ncomp < 1) {
      return;
    }
    const AllToAllWArgs &args = this->get_alltoallw_args<T>(ncomp);

    T null_data = 0;
    T null_data_gathered = 0;

    MPICHK(MPI_Neighbor_alltoallw(
        data != nullptr ? data : &null_data, args.sendcounts.data(),
        args.sdispls.data(), args.sendtypes.data(),
        data_gathered != nullptr ? data_gathered : &null_data_gathered,
        args.recvcounts.data(), args.rdispls.data(), args.recvtypes.data(),
        this->boundary.ncomm));
  }

  /**
   * @param linear_seq_index Linear sequential index for a face geometry index.
   * @returns Geometry ID index for passed index.
   */
  INT get_geom_id_from_seq_index(const INT linear_seq_index);

  /**
   * @param geom_id Linear sequential index for a face geometry index.
   * @returns Geometry ID index for passed index.
   */
  INT get_seq_index_from_geom_id(const INT geom_id);

  /**
   * @returns The device map from geometry IDs to linear sequential indices and
   * the number of linear IDs in the map.
   */
  std::tuple<
      BlockedBinaryNode<INT, INT, NESO_PARTICLES_BLOCKED_BINARY_TREE_WIDTH> *,
      INT>
  get_device_geom_id_to_seq();

  /**
   * Free underlying resource. Should be called collectively on the
   * communicator.
   */
  void free();
};

extern template void
BoundaryMeshInterface::exchange_surface(int *data, const int ncomp,
                                        int *data_gathered);
extern template void
BoundaryMeshInterface::exchange_surface(INT *data, const int ncomp,
                                        INT *data_gathered);
extern template void
BoundaryMeshInterface::exchange_surface(REAL *data, const int ncomp,
                                        REAL *data_gathered);

} // namespace NESO::Particles

#endif
