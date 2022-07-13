#ifndef _NESO_PARTICLES_GLOBAL_MOVE
#define _NESO_PARTICLES_GLOBAL_MOVE

#include <CL/sycl.hpp>
#include <mpi.h>

#include "cell_dat_compression.hpp"
#include "communication.hpp"
#include "compute_target.hpp"
#include "global_move_exchange.hpp"
#include "packing_unpacking.hpp"
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

  BufferShared<int> s_send_ranks;
  BufferShared<int> s_recv_ranks;
  BufferShared<int> s_send_rank_map;
  BufferShared<int> s_pack_cells;
  BufferShared<int> s_pack_layers_src;
  BufferShared<int> s_pack_layers_dst;
  BufferShared<int> s_num_ranks_send;
  BufferShared<int> s_num_ranks_recv;
  BufferShared<int> s_npart_send_recv;
  BufferShared<int> s_send_rank_npart;

  // Reference to the layer compressor on the particle group such that this
  // global move can remove the sent particles
  LayerCompressor &layer_compressor;

public:
  SYCLTarget &sycl_target;

  ~GlobalMove(){};
  GlobalMove(SYCLTarget &sycl_target, LayerCompressor &layer_compressor,
             std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
             std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int)
      : sycl_target(sycl_target), layer_compressor(layer_compressor),
        particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int), particle_packer(sycl_target),
        particle_unpacker(sycl_target), global_move_exchange(sycl_target),
        s_send_ranks(sycl_target, sycl_target.comm_pair.size_parent),
        s_recv_ranks(sycl_target, sycl_target.comm_pair.size_parent),
        s_send_rank_map(sycl_target, sycl_target.comm_pair.size_parent),
        s_pack_cells(sycl_target, 1), s_pack_layers_src(sycl_target, 1),
        s_pack_layers_dst(sycl_target, 1), s_num_ranks_send(sycl_target, 1),
        s_num_ranks_recv(sycl_target, 1), s_npart_send_recv(sycl_target, 1),
        s_send_rank_npart(sycl_target, 1){};
  inline void set_mpi_rank_dat(ParticleDatShPtr<INT> mpi_rank_dat) {
    this->mpi_rank_dat = mpi_rank_dat;
  }

  inline void move() {
    // MPI RMA to inform remote ranks they will recv particles through the
    // global move
    this->global_move_exchange.npart_exchange_init();
    const int comm_size = this->sycl_target.comm_pair.size_parent;
    const int comm_rank = this->sycl_target.comm_pair.rank_parent;

    auto pl_iter_range = this->mpi_rank_dat->get_particle_loop_iter_range();
    auto pl_stride = this->mpi_rank_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->mpi_rank_dat->get_particle_loop_npart_cell();

    this->s_pack_cells.realloc_no_copy(pl_iter_range);
    this->s_pack_layers_src.realloc_no_copy(pl_iter_range);
    this->s_pack_layers_dst.realloc_no_copy(pl_iter_range);

    auto s_send_ranks_ptr = this->s_send_ranks.ptr;
    auto s_recv_ranks_ptr = this->s_recv_ranks.ptr;
    auto s_num_ranks_send_ptr = this->s_num_ranks_send.ptr;
    auto s_num_ranks_recv_ptr = this->s_num_ranks_recv.ptr;
    auto s_send_rank_map_ptr = this->s_send_rank_map.ptr;
    auto s_pack_cells_ptr = this->s_pack_cells.ptr;
    auto s_pack_layers_src_ptr = this->s_pack_layers_src.ptr;
    auto s_pack_layers_dst_ptr = this->s_pack_layers_dst.ptr;
    auto s_npart_send_recv_ptr = this->s_npart_send_recv.ptr;

    // zero the send/recv counts
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<1>(comm_size), [=](sycl::id<1> idx) {
        s_send_ranks_ptr[idx] = 0;
        s_recv_ranks_ptr[idx] = 0;
      });
    });
    // zero the number of ranks involved with send/recv
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<>([=]() {
        s_num_ranks_send_ptr[0] = 0;
        s_num_ranks_recv_ptr[0] = 0;
        s_npart_send_recv_ptr[0] = 0;
      });
    });
    sycl_target.queue.wait_and_throw();
    // loop over all particles - for leaving particles atomically compute the
    // packing layer by incrementing the send count for the report rank and
    // increment the counter for the number of remote ranks to send to
    const INT INT_comm_size = static_cast<INT>(comm_size);
    const INT INT_comm_rank = static_cast<INT>(comm_rank);
    auto d_neso_mpi_rank = this->mpi_rank_dat->cell_dat.device_ptr();

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                const INT cellx = ((INT)idx) / pl_stride;
                const INT layerx = ((INT)idx) % pl_stride;

                if (layerx < pl_npart_cell[cellx]) {
                  const INT owning_rank = d_neso_mpi_rank[cellx][0][layerx];

                  // if rank is valid and not equal to this rank then this
                  // particle is being sent somewhere
                  if (((owning_rank >= 0) && (owning_rank < INT_comm_size)) &&
                      (owning_rank != INT_comm_rank)) {
                    // Increment the counter for the remote rank
                    // reuse the recv ranks array to avoid mallocing more space
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        pack_layer_atomic(s_recv_ranks_ptr[owning_rank]);
                    const int pack_layer = pack_layer_atomic.fetch_add(1);

                    // increment the counter for number of sent particles (all
                    // ranks)
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        send_count_atomic(s_npart_send_recv_ptr[0]);
                    const int send_index = send_count_atomic.fetch_add(1);

                    // store the cell, source layer, packing layer
                    s_pack_cells_ptr[send_index] = static_cast<int>(cellx);
                    s_pack_layers_src_ptr[send_index] =
                        static_cast<int>(layerx);
                    s_pack_layers_dst_ptr[send_index] = pack_layer;

                    // if the packing layer is zero then this is the first
                    // particle found sending to the remote rank -> increment
                    // the number of remote ranks and record this rank.
                    if (pack_layer == 0) {
                      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                       sycl::memory_scope::device>
                          num_ranks_send_atomic(s_num_ranks_send_ptr[0]);
                      const int rank_index = num_ranks_send_atomic.fetch_add(1);
                      s_send_ranks_ptr[rank_index] =
                          static_cast<int>(owning_rank);
                      s_send_rank_map_ptr[owning_rank] = rank_index;
                    }
                  }
                }
              });
        })
        .wait_and_throw();

    const int num_remote_send_ranks = s_num_ranks_send_ptr[0];
    this->s_send_rank_npart.realloc_no_copy(num_remote_send_ranks);
    auto s_send_rank_npart_ptr = this->s_send_rank_npart.ptr;
    const int num_particles_leaving = s_npart_send_recv_ptr[0];

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(num_remote_send_ranks),
                             [=](sycl::id<1> idx) {
                               const int rank = s_send_ranks_ptr[idx];
                               const int npart = s_recv_ranks_ptr[rank];
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

    // pack particles whilst communicating global move information
    this->particle_packer.reset();
    auto global_pack_event = this->particle_packer.pack(
        num_remote_send_ranks, s_send_ranks_ptr, s_send_rank_npart_ptr,
        s_send_rank_map_ptr, num_particles_leaving, s_pack_cells_ptr,
        s_pack_layers_src_ptr, s_pack_layers_dst_ptr, particle_dats_real,
        particle_dats_int);

    // start exchanging global send counts
    this->global_move_exchange.npart_exchange_sendrecv(
        num_remote_send_ranks, this->s_send_ranks, this->s_send_rank_npart);
    this->global_move_exchange.npart_exchange_finalise();

    // allocate space to recv packed particles
    particle_unpacker.reset(this->global_move_exchange.num_remote_recv_ranks,
                            this->global_move_exchange.h_recv_rank_npart,
                            particle_dats_real, particle_dats_int);

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
  }
};

} // namespace NESO::Particles

#endif
