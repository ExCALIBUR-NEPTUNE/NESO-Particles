#ifndef _NESO_PARTICLES_DEPARTING_PARTICLE_IDENTIFICATION
#define _NESO_PARTICLES_DEPARTING_PARTICLE_IDENTIFICATION

#include <mpi.h>

#include "communication.hpp"
#include "compute_target.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"
#include "sycl_typedefs.hpp"

namespace NESO::Particles {

/**
 *  Class to identify which particles are leaving and MPI rank based on which
 *  MPI rank is stored in the NESO_MPI_RANK ParticleDat
 */
class DepartingIdentify {

private:
  SYCLTargetSharedPtr sycl_target;

public:
  /// Disable (implicit) copies.
  DepartingIdentify(const DepartingIdentify &st) = delete;
  /// Disable (implicit) copies.
  DepartingIdentify &operator=(DepartingIdentify const &a) = delete;

  /// Array of unique MPI ranks that particles should be sent to.
  BufferDeviceHost<int> dh_send_ranks;
  /// Array of length equal to the communicator size indicating how many
  /// particles should be sent to each rank.
  BufferDeviceHost<int> dh_send_counts_all_ranks;
  /// Map from MPI ranks to an index that orders the send ranks.
  BufferDeviceHost<int> dh_send_rank_map;
  /// Array that contains the source cells of departing particles.
  BufferDevice<int> d_pack_cells;
  /// Array that contains the source layers (rows) of departing particles.
  BufferDevice<int> d_pack_layers_src;
  /// Array that contains the destination layers in the packing buffer of
  /// departing particles.
  BufferDevice<int> d_pack_layers_dst;
  /// Size one array to accumulate the number of remote ranks particles should
  /// be sent to.
  BufferDeviceHost<int> dh_num_ranks_send;
  /// Size one array to accumulate the total number of leaving particles.
  BufferDeviceHost<int> dh_num_particle_send;

  /// ParticleDat containing MPI ranks.
  ParticleDatSharedPtr<INT> mpi_rank_dat;

  ~DepartingIdentify(){};

  /**
   * Create a new instance of this class.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   */
  DepartingIdentify(SYCLTargetSharedPtr sycl_target)
      : sycl_target(sycl_target),
        dh_send_ranks(sycl_target, sycl_target->comm_pair.size_parent),
        dh_send_counts_all_ranks(sycl_target,
                                 sycl_target->comm_pair.size_parent),
        dh_send_rank_map(sycl_target, sycl_target->comm_pair.size_parent),
        d_pack_cells(sycl_target, 1), d_pack_layers_src(sycl_target, 1),
        d_pack_layers_dst(sycl_target, 1), dh_num_ranks_send(sycl_target, 1),
        dh_num_particle_send(sycl_target, 1){};

  /**
   * Set the ParticleDat that contains particle MPI ranks.
   *
   * @param mpi_rank_dat ParticleDat containing MPI ranks.
   */
  inline void set_mpi_rank_dat(ParticleDatSharedPtr<INT> mpi_rank_dat) {
    this->mpi_rank_dat = mpi_rank_dat;
  }

  /**
   * Identify particles which should be packed and sent to remote MPI ranks.
   * The argument indicates which component of the MPI ranks dat that should be
   * inspected for MPI rank. The intention is that component 0 indicates remote
   * MPI ranks where the particle should be sent through a global communication
   * pattern. Component 1 indicates a remote rank where the particle should be
   * sent through a neighbour based "local" communication pattern. Negative MPI
   * ranks are ignored.
   *
   * @param rank_component Component to inspect for MPI rank.
   */
  inline void identify(const int rank_component = 0);
};

} // namespace NESO::Particles

#endif
