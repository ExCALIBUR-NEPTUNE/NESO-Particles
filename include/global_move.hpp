#ifndef _NESO_PARTICLES_GLOBAL_MOVE
#define _NESO_PARTICLES_GLOBAL_MOVE

#include <CL/sycl.hpp>
#include <mpi.h>

#include "cell_dat_compression.hpp"
#include "communication.hpp"
#include "compute_target.hpp"
#include "departing_particle_identification.hpp"
#include "global_move_exchange.hpp"
#include "packing_unpacking.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/**
 *  Class to move particles on the global mesh.
 */
class GlobalMove {

private:
  std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real;
  std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int;
  ParticleDatShPtr<INT> mpi_rank_dat;

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
  SYCLTarget &sycl_target;

  ~GlobalMove(){};
  /**
   * Construct a new global move instance to move particles between the cells
   * of a MeshHierarchy.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param layer_compressor LayerCompressor to use to compress ParticleDat rows
   * @param particle_dats_real Container of the REAL valued ParticleDats.
   * @param particle_dats_int Container of the INT valued ParticleDats.
   */
  GlobalMove(SYCLTarget &sycl_target, LayerCompressor &layer_compressor,
             std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
             std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int)
      : sycl_target(sycl_target), departing_identify(sycl_target),
        layer_compressor(layer_compressor),
        particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int), particle_packer(sycl_target),
        particle_unpacker(sycl_target), global_move_exchange(sycl_target),
        dh_send_rank_npart(sycl_target, 1){};

  /**
   *  Set the ParticleDat to use for MPI ranks.
   *
   *  @param mpi_rank_dat ParticleDat to use for particle positions.
   */
  inline void set_mpi_rank_dat(ParticleDatShPtr<INT> mpi_rank_dat) {
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
    sycl_target.profile_map.inc("GlobalMove", "move_stage_m3", 1,
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
    sycl_target.profile_map.inc("GlobalMove", "move_stage_m2", 1,
                                profile_elapsed(t0, profile_timestamp()));
    this->sycl_target.queue
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

    sycl_target.profile_map.inc("GlobalMove", "move_stage_m1", 1,
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
    auto global_pack_event = this->particle_packer.pack(
        num_remote_send_ranks, this->dh_send_rank_npart.h_buffer,
        this->departing_identify.dh_send_rank_map, num_particles_leaving,
        this->departing_identify.d_pack_cells,
        this->departing_identify.d_pack_layers_src,
        this->departing_identify.d_pack_layers_dst, this->particle_dats_real,
        this->particle_dats_int);

    auto t0_npart_exchange = profile_timestamp();
    // start exchanging global send counts
    this->global_move_exchange.npart_exchange_sendrecv(
        num_remote_send_ranks, dh_send_ranks,
        this->dh_send_rank_npart.h_buffer);
    this->global_move_exchange.npart_exchange_finalise();
    sycl_target.profile_map.inc(
        "GlobalMove", "npart_exchange_sendrecv", 1,
        profile_elapsed(t0_npart_exchange, profile_timestamp()));

    sycl_target.profile_map.inc("GlobalMove", "move_stage_0", 1,
                                profile_elapsed(t0, profile_timestamp()));

    // allocate space to recv packed particles
    particle_unpacker.reset(this->global_move_exchange.num_remote_recv_ranks,
                            this->global_move_exchange.h_recv_rank_npart,
                            this->particle_dats_real, this->particle_dats_int);

    // TODO can actually start the recv here

    // wait for the local particles to be packed.
    global_pack_event.wait_and_throw();

    sycl_target.profile_map.inc("GlobalMove", "move_stage_1", 1,
                                profile_elapsed(t0, profile_timestamp()));

    // send and recv packed particles from particle_packer.cell_dat to
    // particle_unpacker.h_recv_buffer using h_recv_offsets.
    this->global_move_exchange.exchange_init(this->particle_packer,
                                             this->particle_unpacker);

    sycl_target.profile_map.inc("GlobalMove", "move_stage_2", 1,
                                profile_elapsed(t0, profile_timestamp()));

    // remove the sent particles whilst the communication occurs
    this->layer_compressor.remove_particles(
        num_particles_leaving, this->departing_identify.d_pack_cells.ptr,
        this->departing_identify.d_pack_layers_src.ptr);

    sycl_target.profile_map.inc("GlobalMove", "move_stage_3", 1,
                                profile_elapsed(t0, profile_timestamp()));

    // wait for particle data to be send/recv'd
    this->global_move_exchange.exchange_finalise(this->particle_unpacker);

    // Unpack the recv'd particles
    this->particle_unpacker.unpack(particle_dats_real, particle_dats_int);
    sycl_target.profile_map.inc("GlobalMove", "move", 1,
                                profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO::Particles

#endif
