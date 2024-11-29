#ifndef _NESO_PARTICLES_GLOBAL_MOVE
#define _NESO_PARTICLES_GLOBAL_MOVE

#include <mpi.h>

#include "cell_dat_compression.hpp"
#include "communication.hpp"
#include "compute_target.hpp"
#include "departing_particle_identification.hpp"
#include "global_move_exchange.hpp"
#include "packing_unpacking.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  Class to move particles on the global mesh.
 */
class GlobalMove {

private:
  std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real;
  std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int;
  ParticleDatSharedPtr<INT> mpi_rank_dat;

  ParticlePacker particle_packer;
  ParticleUnpacker particle_unpacker;
  GlobalMoveExchange global_move_exchange;
  DepartingIdentify departing_identify;

  // Reference to the layer compressor on the particle group such that this
  // global move can remove the sent particles
  LayerCompressor &layer_compressor;

  BufferDeviceHost<int> dh_send_rank_npart;

public:
  /// Disable (implicit) copies.
  GlobalMove(const GlobalMove &st) = delete;
  /// Disable (implicit) copies.
  GlobalMove &operator=(GlobalMove const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;

  inline void free() { this->global_move_exchange.free(); }

  ~GlobalMove(){};
  /**
   * Construct a new global move instance to move particles between the cells
   * of a MeshHierarchy.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param layer_compressor LayerCompressor to use to compress ParticleDat rows
   * @param particle_dats_real Container of the REAL valued ParticleDats.
   * @param particle_dats_int Container of the INT valued ParticleDats.
   */
  GlobalMove(
      SYCLTargetSharedPtr sycl_target, LayerCompressor &layer_compressor,
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int)
      : particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int), particle_packer(sycl_target),
        particle_unpacker(sycl_target), global_move_exchange(sycl_target),
        departing_identify(sycl_target), layer_compressor(layer_compressor),
        dh_send_rank_npart(sycl_target, 1), sycl_target(sycl_target){};

  /**
   *  Set the ParticleDat to use for MPI ranks.
   *
   *  @param mpi_rank_dat ParticleDat to use for particle positions.
   */
  inline void set_mpi_rank_dat(ParticleDatSharedPtr<INT> mpi_rank_dat) {
    this->mpi_rank_dat = mpi_rank_dat;
    this->departing_identify.set_mpi_rank_dat(mpi_rank_dat);
  }

  /**
   *  Use the MPI ranks stored in the first component of the MPI rank dat to
   *  move particles between the MPI ranks that own cells in a MeshHierarchy.
   *  Particles are unpacked into cell 0 of the receiving MPI rank.
   */
  inline void move() {
    auto t0 = profile_timestamp();
    // MPI RMA to inform remote ranks they will recv particles through the
    // global move
    this->global_move_exchange.npart_exchange_init();
    sycl_target->profile_map.inc("GlobalMove", "move_stage_a", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    // find particles leaving through the global interface
    this->departing_identify.identify(0);

    const int num_remote_send_ranks =
        this->departing_identify.dh_num_ranks_send.h_buffer.ptr[0];
    this->dh_send_rank_npart.realloc_no_copy(num_remote_send_ranks);
    auto k_send_rank_npart = this->dh_send_rank_npart.d_buffer.ptr;
    const int num_particles_leaving =
        this->departing_identify.dh_num_particle_send.h_buffer.ptr[0];

    auto k_send_ranks = this->departing_identify.dh_send_ranks.d_buffer.ptr;
    auto k_send_counts_all_ranks =
        this->departing_identify.dh_send_counts_all_ranks.d_buffer.ptr;
    sycl_target->profile_map.inc("GlobalMove", "move_stage_b", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(num_remote_send_ranks),
                             [=](sycl::id<1> idx) {
                               const int rank = k_send_ranks[idx];
                               const int npart = k_send_counts_all_ranks[rank];
                               k_send_rank_npart[idx] = npart;
                             });
        })
        .wait_and_throw();
    this->dh_send_rank_npart.device_to_host();

    sycl_target->profile_map.inc("GlobalMove", "move_stage_c", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    // We now have:
    // 1) n particles to send
    // 2) array length n of source cells
    // 3) array length n of source layers (the rows in the source cells)
    // 4) array length n of packing layers (the index in the packing buffer for
    //    each particle to send)
    // 5) m destination MPI ranks to send to
    // 6) array length m of destination ranks
    // 7) array length m of particle counts to send to each destination ranks

    auto &dh_send_ranks = this->departing_identify.dh_send_ranks;

    // pack particles whilst communicating global move information
    this->particle_packer.reset();
    sycl_target->profile_map.inc("GlobalMove", "move_stage_d", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    auto global_pack_event = this->particle_packer.pack(
        num_remote_send_ranks, this->dh_send_rank_npart.h_buffer,
        this->departing_identify.dh_send_rank_map, num_particles_leaving,
        this->departing_identify.d_pack_cells,
        this->departing_identify.d_pack_layers_src,
        this->departing_identify.d_pack_layers_dst, this->particle_dats_real,
        this->particle_dats_int);

    sycl_target->profile_map.inc("GlobalMove", "move_stage_e", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // start exchanging global send counts
    this->global_move_exchange.npart_exchange_sendrecv(
        num_remote_send_ranks, dh_send_ranks,
        this->dh_send_rank_npart.h_buffer);

    sycl_target->profile_map.inc("GlobalMove", "move_stage_f", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    this->global_move_exchange.npart_exchange_finalise();

    sycl_target->profile_map.inc("GlobalMove", "move_stage_g", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // allocate space to recv packed particles
    particle_unpacker.reset(this->global_move_exchange.num_remote_recv_ranks,
                            this->global_move_exchange.h_recv_rank_npart,
                            this->particle_dats_real, this->particle_dats_int);

    sycl_target->profile_map.inc("GlobalMove", "move_stage_h", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    // TODO can actually start the recv here

    // wait for the local particles to be packed.
    global_pack_event.wait_and_throw();

    sycl_target->profile_map.inc("GlobalMove", "move_stage_i", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // send and recv packed particles from particle_packer.cell_dat to
    // particle_unpacker.h_recv_buffer using h_recv_offsets.
    this->global_move_exchange.exchange_init(this->particle_packer,
                                             this->particle_unpacker);

    sycl_target->profile_map.inc("GlobalMove", "move_stage_j", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // remove the sent particles whilst the communication occurs
    this->layer_compressor.remove_particles(
        num_particles_leaving, this->departing_identify.d_pack_cells.ptr,
        this->departing_identify.d_pack_layers_src.ptr);

    sycl_target->profile_map.inc("GlobalMove", "move_stage_k", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    // wait for particle data to be send/recv'd
    this->global_move_exchange.exchange_finalise(this->particle_unpacker);

    // Unpack the recv'd particles
    this->particle_unpacker.unpack(particle_dats_real, particle_dats_int);
    sycl_target->profile_map.inc("GlobalMove", "move", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    sycl_target->profile_map.inc("GlobalMove", "send_count",
                                 (INT)num_particles_leaving, 0.0);
  }
};

} // namespace NESO::Particles

#endif
