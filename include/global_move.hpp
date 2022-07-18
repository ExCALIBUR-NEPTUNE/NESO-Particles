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

  BufferShared<int> s_send_rank_npart;

public:
  SYCLTarget &sycl_target;

  ~GlobalMove(){};
  GlobalMove(SYCLTarget &sycl_target, LayerCompressor &layer_compressor,
             std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
             std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int)
      : sycl_target(sycl_target), departing_identify(sycl_target),
        layer_compressor(layer_compressor),
        particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int), particle_packer(sycl_target),
        particle_unpacker(sycl_target), global_move_exchange(sycl_target),
        s_send_rank_npart(sycl_target, 1){};
  inline void set_mpi_rank_dat(ParticleDatShPtr<INT> mpi_rank_dat) {
    this->mpi_rank_dat = mpi_rank_dat;
    this->departing_identify.set_mpi_rank_dat(mpi_rank_dat);
  }

  inline void move() {
    auto t0 = profile_timestamp();
    // MPI RMA to inform remote ranks they will recv particles through the
    // global move
    this->global_move_exchange.npart_exchange_init();

    // find particles leaving through the global interface
    this->departing_identify.identify(0);

    const int num_remote_send_ranks =
        this->departing_identify.s_num_ranks_send.ptr[0];
    this->s_send_rank_npart.realloc_no_copy(num_remote_send_ranks);
    auto s_send_rank_npart_ptr = this->s_send_rank_npart.ptr;
    const int num_particles_leaving =
        this->departing_identify.s_npart_send_recv.ptr[0];

    auto s_send_ranks_ptr = this->departing_identify.s_send_ranks.ptr;
    auto s_send_counts_all_ranks_ptr =
        this->departing_identify.s_send_counts_all_ranks.ptr;

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(num_remote_send_ranks), [=](sycl::id<1> idx) {
                const int rank = s_send_ranks_ptr[idx];
                const int npart = s_send_counts_all_ranks_ptr[rank];
                s_send_rank_npart_ptr[idx] = npart;
              });
        })
        .wait_and_throw();

    // We now have:
    // 1) n particles to send
    // 2) array length n of source cells
    // 3) array length n of source layers (the rows in the source cells)
    // 4) array length n of packing layers (the index in the packing buffer for
    //    each particle to send)
    // 5) m destination MPI ranks to send to
    // 6) array length m of destination ranks
    // 7) array length m of particle counts to send to each destination ranks

    auto s_pack_cells_ptr = this->departing_identify.s_pack_cells.ptr;
    auto s_pack_layers_src_ptr = this->departing_identify.s_pack_layers_src.ptr;
    auto s_pack_layers_dst_ptr = this->departing_identify.s_pack_layers_dst.ptr;
    auto s_send_rank_map_ptr = this->departing_identify.s_send_rank_map.ptr;
    auto &s_send_ranks = this->departing_identify.s_send_ranks;

    // pack particles whilst communicating global move information
    this->particle_packer.reset();
    auto global_pack_event = this->particle_packer.pack(
        num_remote_send_ranks, s_send_rank_npart_ptr, s_send_rank_map_ptr,
        num_particles_leaving, s_pack_cells_ptr, s_pack_layers_src_ptr,
        s_pack_layers_dst_ptr, this->particle_dats_real,
        this->particle_dats_int);

    // start exchanging global send counts
    this->global_move_exchange.npart_exchange_sendrecv(
        num_remote_send_ranks, s_send_ranks, this->s_send_rank_npart);
    this->global_move_exchange.npart_exchange_finalise();

    // allocate space to recv packed particles
    particle_unpacker.reset(this->global_move_exchange.num_remote_recv_ranks,
                            this->global_move_exchange.h_recv_rank_npart,
                            this->particle_dats_real, this->particle_dats_int);

    // TODO can actually start the recv here

    // wait for the local particles to be packed.
    global_pack_event.wait_and_throw();

    // send and recv packed particles from particle_packer.cell_dat to
    // particle_unpacker.h_recv_buffer using h_recv_offsets.
    this->global_move_exchange.exchange_init(this->particle_packer,
                                             this->particle_unpacker);

    // remove the sent particles whilst the communication occurs
    this->layer_compressor.remove_particles(
        num_particles_leaving, s_pack_cells_ptr, s_pack_layers_src_ptr);

    // wait for particle data to be send/recv'd
    this->global_move_exchange.exchange_finalise(this->particle_unpacker);

    // Unpack the recv'd particles
    this->particle_unpacker.unpack(particle_dats_real, particle_dats_int);
    sycl_target.profile_map.inc("GlobalMove", "Move", 1,
                                profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO::Particles

#endif
