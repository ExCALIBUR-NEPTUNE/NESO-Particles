#ifndef _NESO_PARTICLES_PACKING_UNPACKING
#define _NESO_PARTICLES_PACKING_UNPACKING

#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <stack>
#include <string>

#include "compute_target.hpp"
#include "particle_dat.hpp"
#include "particle_group_pointer_map.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  Class to pack particle data to send using MPI operations.
 */
class ParticlePacker {
private:
  int num_dats_real = 0;
  int num_dats_int = 0;
  BufferDeviceHost<REAL *const *const *> dh_particle_dat_ptr_real;
  BufferDeviceHost<INT *const *const *> dh_particle_dat_ptr_int;
  BufferDeviceHost<int> dh_particle_dat_ncomp_real;
  BufferDeviceHost<int> dh_particle_dat_ncomp_int;
  bool device_aware_mpi_enabled;

  size_t particle_size(
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int);

  void get_particle_dat_info(
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int);

  /// Required length of the send buffer.
  INT required_send_buffer_length;
  /// Vector of pointers for packed cells on device.
  BufferHost<char *> h_send_pointers;
  /// CellDat used to pack the particles to be sent to each remote rank on the
  // device.
  CellDat<char> cell_dat;
  /// Host buffer to copy packed data to before sending using MPI routines.
  BufferHost<char> h_send_buffer;
  /// Vector of offsets to index into the host send buffer.
  BufferHost<INT> h_send_offsets;

public:
  /// Disable (implicit) copies.
  ParticlePacker(const ParticlePacker &st) = delete;
  /// Disable (implicit) copies.
  ParticlePacker &operator=(ParticlePacker const &a) = delete;

  /// Number of bytes required per particle packed.
  int num_bytes_per_particle;
  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  ~ParticlePacker(){};
  /**
   * Construct a particle packing object on a compute device.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   */
  ParticlePacker(SYCLTargetSharedPtr sycl_target)
      : dh_particle_dat_ptr_real(sycl_target, 1),
        dh_particle_dat_ptr_int(sycl_target, 1),
        dh_particle_dat_ncomp_real(sycl_target, 1),
        dh_particle_dat_ncomp_int(sycl_target, 1),
        device_aware_mpi_enabled(NESO::Particles::device_aware_mpi_enabled()),
        h_send_pointers(sycl_target, 1),
        cell_dat(sycl_target, sycl_target->comm_pair.size_parent, 1),
        h_send_buffer(sycl_target, 1), h_send_offsets(sycl_target, 1),
        sycl_target(sycl_target) {}
  /**
   *  Reset the instance before packing particle data.
   */
  inline void reset() {}

  /**
   * @returns Array of size num_remote_send_ranks containing pointers to host
   * or device memory depending on if device aware MPI is enabled.
   */
  char **get_packed_pointers(const int num_remote_send_ranks,
                             const int *h_send_rank_npart_ptr);

  /**
   *  Pack particle data on the device.
   *
   * @param num_remote_send_ranks Number of remote ranks involved in send.
   * @param h_send_rank_npart Host buffer holding the number of particles to be
   * sent to each remote rank.
   * @param dh_send_rank_map Maps MPI ranks to the cell in the packing CellDat.
   * @param num_particles_leaving Total number of particles to pack.
   * @param d_pack_cells BufferDevice holding the cells of particles to pack.
   * @param d_pack_layers_src BufferDevice holding the layers(rows) of particles
   * to pack.
   * @param d_pack_layers_dst BufferDevice holding the destination layers in
   * the packing CellDat.
   * @param particle_dats_real Container of REAL ParticleDat instances to pack.
   * @param particle_dats_int Container of INT ParticleDat instances to pack.
   * @param rank_component Component of MPI rank ParticleDat to inspect for
   * destination MPI rank.
   * @returns sycl::event to wait on for packing completion.
   */
  sycl::event
  pack(const int num_remote_send_ranks, BufferHost<int> &h_send_rank_npart,
       BufferDeviceHost<int> &dh_send_rank_map, const int num_particles_leaving,
       BufferDevice<int> &d_pack_cells, BufferDevice<int> &d_pack_layers_src,
       BufferDevice<int> &d_pack_layers_dst,
       std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
       std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int,
       const int rank_component = 0);

  /**
   *  Pack particle data on the device.
   *
   * @param num_remote_send_ranks Number of remote ranks involved in send.
   * @param h_send_rank_npart Host buffer holding the number of particles to be
   * sent to each remote rank.
   * @param dh_send_rank_map Maps MPI ranks to the cell in the packing CellDat.
   * @param num_particles_leaving Total number of particles to pack.
   * @param d_pack_cells BufferDevice holding the cells of particles to pack.
   * @param d_pack_layers_src BufferDevice holding the layers(rows) of particles
   * to pack.
   * @param d_pack_layers_dst BufferDevice holding the destination layers in
   * the packing CellDat.
   * @param particle_group_pointer_map ParticleGroupPointerMap describing
   * ParticleDat information.
   * @param event_stack EventStack to push events onto to wait on.
   * @param rank_component Component of MPI rank ParticleDat to inspect for
   * destination MPI rank.
   */
  void pack(const int num_remote_send_ranks, BufferHost<int> &h_send_rank_npart,
            BufferDeviceHost<int> &dh_send_rank_map,
            const int num_particles_leaving, BufferDevice<int> &d_pack_cells,
            BufferDevice<int> &d_pack_layers_src,
            BufferDevice<int> &d_pack_layers_dst,
            ParticleGroupPointerMapSharedPtr particle_group_pointer_map,
            EventStack &event_stack, const int rank_component = 0);
};

/**
 * Class to unpack particle data which was packed using the ParticlePacker.
 */
class ParticleUnpacker {
private:
  int num_dats_real = 0;
  int num_dats_int = 0;

  BufferDeviceHost<REAL ***> dh_particle_dat_ptr_real;
  BufferDeviceHost<INT ***> dh_particle_dat_ptr_int;
  BufferDeviceHost<int> dh_particle_dat_ncomp_real;
  BufferDeviceHost<int> dh_particle_dat_ncomp_int;
  BufferDevice<char> d_recv_buffer;

  bool device_aware_mpi_enabled;

  size_t particle_size(
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int);

  void get_particle_dat_info(
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int);

  /// Pointers in which to recv data
  BufferHost<char *> h_recv_pointers;
  /// Host buffer to receive particle data into from MPI operations.
  BufferHost<char> h_recv_buffer;
  /// Offsets into the recv buffer for each remote rank that will send to this
  // rank.
  BufferHost<INT> h_recv_offsets;

public:
  /// Disable (implicit) copies.
  ParticleUnpacker(const ParticleUnpacker &st) = delete;
  /// Disable (implicit) copies.
  ParticleUnpacker &operator=(ParticleUnpacker const &a) = delete;

  /// Number of particles expected in the next/current recv operation.
  int npart_recv;
  /// Number of bytes per particle.
  int num_bytes_per_particle;
  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;

  ~ParticleUnpacker(){};

  /**
   * Construct an unpacking object.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   */
  ParticleUnpacker(SYCLTargetSharedPtr sycl_target)
      : dh_particle_dat_ptr_real(sycl_target, 1),
        dh_particle_dat_ptr_int(sycl_target, 1),
        dh_particle_dat_ncomp_real(sycl_target, 1),
        dh_particle_dat_ncomp_int(sycl_target, 1),
        d_recv_buffer(sycl_target, 1),
        device_aware_mpi_enabled(NESO::Particles::device_aware_mpi_enabled()),
        h_recv_pointers(sycl_target, 1), h_recv_buffer(sycl_target, 1),
        h_recv_offsets(sycl_target, 1), sycl_target(sycl_target){};

  /**
   * @returns Host or device pointers for locations in which to recv packed
   * particles from each MPI rank.
   */
  char **get_recv_pointers(const int num_remote_recv_ranks);

  /**
   *  Reset the unpacker ready to unpack received particles.
   *
   *  @param num_remote_recv_ranks Number of MPI ranks that will send to this
   * rank.
   *  @param h_recv_rank_npart Number of particles each rank will send to this
   * rank.
   *  @param particle_dats_real Container of REAL ParticleDat instances to
   * unpack.
   *  @param particle_dats_int Container of INT ParticleDat instances to unpack.
   */
  void
  reset(const int num_remote_recv_ranks, BufferHost<int> &h_recv_rank_npart,
        std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
        std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int);

  /**
   * Unpack the recv buffer into the particle group. Particles unpack into cell
   * 0.
   *
   * @param particle_dats_real Container of REAL ParticleDat instances to
   * unpack.
   * @param particle_dats_int Container of INT ParticleDat instances to unpack.
   */
  void
  unpack(std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
         std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int);
};

} // namespace NESO::Particles

#endif
