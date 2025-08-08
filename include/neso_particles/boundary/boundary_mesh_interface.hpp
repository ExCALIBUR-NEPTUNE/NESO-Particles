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
    // Neighbour comm for boundary exchanges with edges reversed from ncomm.
    MPI_Comm rncomm{MPI_COMM_NULL};
    // Map from remote MPI ranks to geometry ids that those remote ranks own but
    // this rank holds a contribution for.
    std::map<int, std::set<int>> map_recv_rank_to_geom_ids;
    // Map from remote ranks which hold copies of geometry ids this rank owns
    // and the geometry objects that they hold a contribution for.
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
    // Device side geometry ids of the incoming data.
    std::shared_ptr<BufferDevice<int>> d_incoming_geom_ids;
    // Geometry ids of the outgoing data
    std::vector<int> outgoing_geom_ids;
    // Counter to place an ordering on face geom ids (outgoing)
    INT geom_counter{0};
    // Map from linear sequential index to geom id (outgoing).
    std::map<INT, INT> map_linear_index_to_geom_id;
    // Map from geom id to linear sequential index (outgoing).
    std::map<INT, INT> map_geom_id_to_linear_index;
    // Device map from geom id to linear sequential index (outgoing).
    std::shared_ptr<BlockedBinaryTree<INT, INT>> d_map_geom_id_to_linear_index;
    // Map from sequential linear index to packing index, -1 if not packed.
    std::shared_ptr<BufferDevice<int>> d_outgoing_pack_index;
    // Is device aware MPI enabled?
    bool device_aware_mpi{false};
    // Owned face geometry IDs (map from linear index to geometry ID)
    std::vector<INT> owned_geom_ids;
    // Map from an owned geometry ID to its linear index.
    std::map<INT, INT> map_owned_geom_id_to_linear_index;
    // Device map from owned geometry ID to linear index.
    std::shared_ptr<BlockedBinaryTree<INT, INT>>
        d_map_owned_geom_id_to_linear_index;

    // Device map for packing geometry for the reverse (evaluation) direction
    std::shared_ptr<BufferDevice<int>> d_reverse_outgoing_pack_index;
    // Device map for unpacking geometry for the reverse (evaluate) direction
    std::shared_ptr<BufferDevice<int>> d_reverse_incoming_unpack_index;
    // Number of geometery ids each remote rank will send to this rank. Ordering
    // is defined by the MPI graph.
    std::vector<int> reverse_incoming_geom_counts;
    // Number of geometery ids to send to each remote rank. Ordering is defined
    // by the MPI graph.
    std::vector<int> reverse_outgoing_geom_counts;

    std::map<std::pair<std::type_index, int>, AllToAllWArgs>
        map_typencomp_alltoallwargs;
    std::map<std::pair<std::type_index, int>, AllToAllWArgs>
        reverse_map_typencomp_alltoallwargs;

    struct {
      int indegree{0};
      int outdegree{0};
      std::vector<int> sources;
      std::vector<int> destinations;
      std::vector<int> reverse_sources;
      std::vector<int> reverse_destinations;
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

  template <typename T>
  const AllToAllWArgs &get_reverse_alltoallw_args(const int ncomp) {
    if (!this->boundary.reverse_map_typencomp_alltoallwargs.count(
            {typeid(T), ncomp})) {

      AllToAllWArgs args;
      // N.B. these in/outdegrees are for the projection direction, i.e. not the
      // reverse direction.
      const auto outdegree = this->boundary.graph.outdegree;
      const auto indegree = this->boundary.graph.indegree;

      args.sendcounts.resize(std::max(1, indegree));
      args.recvcounts.resize(std::max(1, outdegree));
      args.sendtypes.resize(std::max(1, indegree));
      args.recvtypes.resize(std::max(1, outdegree));
      args.sdispls.resize(std::max(1, indegree));
      args.rdispls.resize(std::max(1, outdegree));

      MPI_Aint offset = 0;
      for (int ix = 0; ix < indegree; ix++) {
        const int geom_count = this->boundary.reverse_outgoing_geom_counts[ix];
        args.sendcounts[ix] = geom_count * ncomp;
        args.sdispls[ix] = offset;
        offset += geom_count * ncomp * sizeof(T);
      }
      offset = 0;
      for (int ix = 0; ix < outdegree; ix++) {
        const int geom_count = this->boundary.reverse_incoming_geom_counts[ix];
        args.recvcounts[ix] = geom_count * ncomp;
        args.rdispls[ix] = offset;
        offset += geom_count * ncomp * sizeof(T);
      }
      std::fill(args.sendtypes.begin(), args.sendtypes.end(),
                map_ctype_mpi_type<T>());
      std::fill(args.recvtypes.begin(), args.recvtypes.end(),
                map_ctype_mpi_type<T>());

      this->boundary.reverse_map_typencomp_alltoallwargs[{typeid(T), ncomp}] =
          args;
    }
    return this->boundary.reverse_map_typencomp_alltoallwargs.at(
        {typeid(T), ncomp});
  }

public:
  BoundaryMeshInterface() = default;

  /**
   * This constructor is collective on the communicator.
   *
   * @param comm MPI communicator for mesh.
   * @param sycl_target Compute device for device maps.
   * @param owned_face_cells The geometry IDs of boundary cells which this MPI
   * ranks owns.
   */
  BoundaryMeshInterface(MPI_Comm comm, SYCLTargetSharedPtr sycl_target,
                        const std::vector<INT> &owned_face_cells);

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
   * Send data from each rank to the owning rank for the data. Collective on the
   * communicator.
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
    if (ncomp < 1 || this->boundary.ncomm == MPI_COMM_NULL) {
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
   * Send data from the owning rank to remote ranks that require copies of the
   * data. Collective on the communicator.
   *
   * @param[in] data Data to send from owning ranks. num_geoms x ncomp sized
   * array ordered by component fastest followed by geometry index. Geometry
   * object id and ordering is defined by the linear order for owned geoms.
   * @param[in] ncomp Number of components to send per geometry object.
   * @param[in, out] data_gathered Output data num_geoms x ncomp sized array
   * ordered by component fastest followed by geometry index.
   */
  template <typename T>
  void reverse_exchange_surface(T *data, const int ncomp, T *data_gathered) {
    if (ncomp < 1 || this->boundary.rncomm == MPI_COMM_NULL) {
      return;
    }
    const AllToAllWArgs &args = this->get_reverse_alltoallw_args<T>(ncomp);

    T null_data = 0;
    T null_data_gathered = 0;

    MPICHK(MPI_Neighbor_alltoallw(
        data != nullptr ? data : &null_data, args.sendcounts.data(),
        args.sdispls.data(), args.sendtypes.data(),
        data_gathered != nullptr ? data_gathered : &null_data_gathered,
        args.recvcounts.data(), args.rdispls.data(), args.recvtypes.data(),
        this->boundary.rncomm));
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
   * @returns The number of local faces plus remote faces that could be hit by
   * particles.
   */
  int get_num_intersection_geoms() const;

  /**
   * @returns The total number of exported geoms across all ranks (including the
   * same geom sent to different ranks).
   */
  int get_total_num_exported_geoms() const;

  /**
   * Free underlying resource. Should be called collectively on the
   * communicator.
   */
  void free();

protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  template <typename T>
  [[nodiscard]] sycl::event exchange_from_device_pack(T *d_src, const int ncomp,
                                                      T *k_packed_out) {

    if (this->boundary.geom_counter) {
      auto sycl_target = this->boundary.sycl_target;
      auto *k_outgoing_pack_index = this->boundary.d_outgoing_pack_index->ptr;

      auto e0 = sycl_target->queue.parallel_for(
          sycl::range<2>(this->boundary.geom_counter, ncomp),
          [=](sycl::item<2> idx) {
            const std::size_t idx_geom = idx.get_id(0);
            const std::size_t idx_component = idx.get_id(1);
            const int geom_dst = k_outgoing_pack_index[idx_geom];
            k_packed_out[geom_dst * ncomp + idx_component] =
                d_src[idx_geom * ncomp + idx_component];
          });

      return e0;
    } else {
      return sycl::event{};
    }
  }

  template <typename T>
  [[nodiscard]] sycl::event
  exchange_from_device_unpack(T *k_packed_in, const int ncomp, T *d_dst) {
    const std::size_t num_incoming_geom_ids =
        this->boundary.incoming_geom_ids.size();

    if (num_incoming_geom_ids) {
      auto sycl_target = this->boundary.sycl_target;

      int *k_incoming_geom_ids = this->boundary.d_incoming_geom_ids->ptr;
      auto *k_map_owned_geom_id_to_linear_index =
          this->boundary.d_map_owned_geom_id_to_linear_index->root;

      NESOASSERT(k_map_owned_geom_id_to_linear_index != nullptr,
                 "This map should contain geometry objects.");

      return sycl_target->queue.parallel_for(
          sycl::range<2>(num_incoming_geom_ids, ncomp), [=](sycl::item<2> idx) {
            const INT *linear_index = nullptr;
            const INT incoming_index = idx.get_id(0);
            const INT gid = k_incoming_geom_ids[incoming_index];
            const INT cx = idx.get_id(1);
            k_map_owned_geom_id_to_linear_index->get(gid, &linear_index);
            atomic_fetch_add(d_dst + ncomp * (*linear_index) + cx,
                             k_packed_in[incoming_index * ncomp + cx]);
          });
    } else {
      return sycl::event{};
    }
  }

  template <typename T>
  [[nodiscard]] sycl::event reverse_exchange_from_device_pack(T *d_src,
                                                              const int ncomp,
                                                              T *k_packed_out) {

    const std::size_t num_incoming_geom_ids =
        this->boundary.incoming_geom_ids.size();
    // If there are incoming geoms for deposition then there are outgoing geoms
    // for evaluation.
    if (num_incoming_geom_ids) {

      auto sycl_target = this->boundary.sycl_target;
      auto *k_pack_index = this->boundary.d_reverse_outgoing_pack_index->ptr;

      auto e0 = sycl_target->queue.parallel_for(
          sycl::range<2>(this->boundary.d_reverse_outgoing_pack_index->size,
                         ncomp),
          [=](sycl::item<2> idx) {
            const std::size_t idx_linear = idx.get_id(0);
            const std::size_t idx_component = idx.get_id(1);
            const int idx_map = k_pack_index[idx_linear];
            k_packed_out[idx_linear * ncomp + idx_component] =
                d_src[idx_map * ncomp + idx_component];
          });

      return e0;
    } else {
      return sycl::event{};
    }
  }

  template <typename T>
  [[nodiscard]] sycl::event reverse_exchange_from_device_unpack(T *k_packed_in,
                                                                const int ncomp,
                                                                T *d_dst) {

    if (this->boundary.geom_counter) {
      auto sycl_target = this->boundary.sycl_target;
      auto *k_unpack_index =
          this->boundary.d_reverse_incoming_unpack_index->ptr;

      return sycl_target->queue.parallel_for(
          sycl::range<2>(this->boundary.d_reverse_incoming_unpack_index->size,
                         ncomp),
          [=](sycl::item<2> idx) {
            const std::size_t idx_linear = idx.get_id(0);
            const std::size_t idx_component = idx.get_id(1);
            const int idx_map = k_unpack_index[idx_linear];
            d_dst[idx_map * ncomp + idx_component] =
                k_packed_in[idx_linear * ncomp + idx_component];
          });
    } else {
      return sycl::event{};
    }
  }

public:
  /**
   * Pack and exchange DOFs supplied in a device buffer. DOFs for each geometry
   * object are combined using addition. This is intended for the
   * deposition/projection direction. Collective on the communicator.
   *
   * @param d_src Source data on device.
   * @param ncomp Number of values that will be exchanged per geometry object.
   * @param d_dst Destination buffer on device. This buffer will be incremented
   * with the incoming values.
   */
  template <typename T>
  void exchange_from_device(T *d_src, const int ncomp, T *d_dst) {
    const std::size_t size_tmp_out =
        this->boundary.total_num_outgoing_geoms * ncomp;
    const std::size_t size_tmp_in =
        this->boundary.total_num_incoming_geoms * ncomp;

    auto sycl_target = this->boundary.sycl_target;

    auto d_packed_out =
        get_resource<BufferDevice<T>, ResourceStackInterfaceBufferDevice<T>>(
            sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<T>{},
            sycl_target);
    d_packed_out->realloc_no_copy(size_tmp_out);
    T *k_packed_out = d_packed_out->ptr;

    auto e0 = this->exchange_from_device_pack(d_src, ncomp, k_packed_out);

    auto d_packed_in =
        get_resource<BufferDevice<T>, ResourceStackInterfaceBufferDevice<T>>(
            sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<T>{},
            sycl_target);
    d_packed_in->realloc_no_copy(size_tmp_in);
    T *k_packed_in = d_packed_in->ptr;

    // Wait for the packing to finish.
    e0.wait_and_throw();

    std::vector<T> h_packed_out;
    std::vector<T> h_packed_in;
    T *m_packed_out = k_packed_out;
    T *m_packed_in = k_packed_in;
    if (!this->boundary.device_aware_mpi) {
      h_packed_out.resize(size_tmp_out);
      h_packed_in.resize(size_tmp_in);
      m_packed_out = h_packed_out.data();
      m_packed_in = h_packed_in.data();
      sycl_target->queue
          .memcpy(m_packed_out, k_packed_out, size_tmp_out * sizeof(T))
          .wait_and_throw();
    }
    this->exchange_surface(m_packed_out, ncomp, m_packed_in);

    h_packed_out = std::vector<T>{};
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<T>{}, d_packed_out);

    if (!this->boundary.device_aware_mpi) {
      sycl_target->queue
          .memcpy(k_packed_in, m_packed_in, size_tmp_in * sizeof(T))
          .wait_and_throw();
    }
    h_packed_in = std::vector<T>{};

    this->exchange_from_device_unpack(k_packed_in, ncomp, d_dst)
        .wait_and_throw();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<T>{}, d_packed_in);
  }

  /**
   * Pack and exchange DOFs supplied in a device buffer. DOFs for each geometry
   * object are combined using addition. This is intended for the
   * evaluation direction. Collective on the communicator.
   *
   * @param d_src Source data on device.
   * @param ncomp Number of values that will be exchanged per geometry object.
   * @param d_dst Destination buffer on device. This buffer will be replaced
   * with the incoming values.
   */
  template <typename T>
  void reverse_exchange_from_device(T *d_src, const int ncomp, T *d_dst) {

    auto sycl_target = this->boundary.sycl_target;
    const int packed_size =
        this->boundary.d_reverse_outgoing_pack_index
            ? this->boundary.d_reverse_outgoing_pack_index->size * ncomp
            : 0;
    const int unpacked_size =
        this->boundary.d_reverse_incoming_unpack_index
            ? this->boundary.d_reverse_incoming_unpack_index->size * ncomp
            : 0;

    auto d_packed =
        get_resource<BufferDevice<T>, ResourceStackInterfaceBufferDevice<T>>(
            sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<T>{},
            sycl_target);
    d_packed->realloc_no_copy(packed_size);
    auto d_recv_packed =
        get_resource<BufferDevice<T>, ResourceStackInterfaceBufferDevice<T>>(
            sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<T>{},
            sycl_target);
    d_recv_packed->realloc_no_copy(unpacked_size);

    this->reverse_exchange_from_device_pack(d_src, ncomp, d_packed->ptr)
        .wait_and_throw();

    T *m_packed = d_packed->ptr;
    T *m_recvd_packed = d_recv_packed->ptr;

    std::vector<T> h_packed;
    std::vector<T> h_recv_packed;

    if (!this->boundary.device_aware_mpi) {
      h_packed.resize(packed_size);
      h_recv_packed.resize(unpacked_size);
      m_packed = h_packed.data();
      m_recvd_packed = h_recv_packed.data();
      if (packed_size > 0) {
        sycl_target->queue
            .memcpy(m_packed, d_packed->ptr, packed_size * sizeof(T))
            .wait_and_throw();
      }
    }

    reverse_exchange_surface(m_packed, ncomp, m_recvd_packed);

    if (!this->boundary.device_aware_mpi) {
      if (unpacked_size > 0) {
        sycl_target->queue
            .memcpy(d_recv_packed->ptr, m_recvd_packed,
                    unpacked_size * sizeof(T))
            .wait_and_throw();
      }
    }

    this->reverse_exchange_from_device_unpack(m_recvd_packed, ncomp, d_dst)
        .wait_and_throw();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<T>{}, d_recv_packed);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<T>{}, d_packed);
  }
};

typedef std::shared_ptr<BoundaryMeshInterface> BoundaryMeshInterfaceSharedPtr;

extern template void
BoundaryMeshInterface::exchange_surface(int *data, const int ncomp,
                                        int *data_gathered);
extern template void
BoundaryMeshInterface::exchange_surface(INT *data, const int ncomp,
                                        INT *data_gathered);
extern template void
BoundaryMeshInterface::exchange_surface(REAL *data, const int ncomp,
                                        REAL *data_gathered);
extern template void
BoundaryMeshInterface::exchange_from_device(REAL *d_src, const int ncomp,
                                            REAL *d_dst);
extern template void
BoundaryMeshInterface::exchange_from_device(INT *d_src, const int ncomp,
                                            INT *d_dst);
extern template sycl::event
BoundaryMeshInterface::exchange_from_device_pack(REAL *k_packed_in,
                                                 const int ncomp, REAL *d_dst);
extern template sycl::event
BoundaryMeshInterface::exchange_from_device_pack(INT *k_packed_in,
                                                 const int ncomp, INT *d_dst);
extern template sycl::event BoundaryMeshInterface::exchange_from_device_unpack(
    REAL *k_packed_in, const int ncomp, REAL *d_dst);
extern template sycl::event
BoundaryMeshInterface::exchange_from_device_unpack(INT *k_packed_in,
                                                   const int ncomp, INT *d_dst);

} // namespace NESO::Particles

#endif
