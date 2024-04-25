#ifndef _NESO_PARTICLES_DEPARTING_PARTICLE_IDENTIFICATION_IMPL_H_
#define _NESO_PARTICLES_DEPARTING_PARTICLE_IDENTIFICATION_IMPL_H_

#include "departing_particle_identification.hpp"
#include "loop/particle_loop.hpp"

namespace NESO::Particles {

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
inline void DepartingIdentify::identify(const int rank_component) {
  auto t0 = profile_timestamp();

  const int comm_size = this->sycl_target->comm_pair.size_parent;
  const int comm_rank = this->sycl_target->comm_pair.rank_parent;

  NESOASSERT(this->mpi_rank_dat.get() != nullptr,
             "MPI rank dat is not defined");

  const auto npart_local = this->mpi_rank_dat->get_npart_local();
  this->d_pack_cells.realloc_no_copy(npart_local);
  this->d_pack_layers_src.realloc_no_copy(npart_local);
  this->d_pack_layers_dst.realloc_no_copy(npart_local);

  auto k_send_ranks = this->dh_send_ranks.d_buffer.ptr;
  auto k_send_counts_all_ranks = this->dh_send_counts_all_ranks.d_buffer.ptr;
  auto k_num_ranks_send = this->dh_num_ranks_send.d_buffer.ptr;
  auto k_send_rank_map = this->dh_send_rank_map.d_buffer.ptr;
  auto k_pack_cells = this->d_pack_cells.ptr;
  auto k_pack_layers_src = this->d_pack_layers_src.ptr;
  auto k_pack_layers_dst = this->d_pack_layers_dst.ptr;
  auto k_num_particle_send = this->dh_num_particle_send.d_buffer.ptr;

  // zero the send/recv counts
  this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<1>(comm_size), [=](sycl::id<1> idx) {
      k_send_ranks[idx] = 0;
      k_send_counts_all_ranks[idx] = 0;
    });
  });
  // zero the number of ranks involved with send/recv
  this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<>([=]() {
      k_num_ranks_send[0] = 0;
      k_num_particle_send[0] = 0;
    });
  });
  sycl_target->queue.wait_and_throw();
  // loop over all particles - for leaving particles atomically compute the
  // packing layer by incrementing the send count for the report rank and
  // increment the counter for the number of remote ranks to send to
  const INT INT_comm_size = static_cast<INT>(comm_size);
  const INT INT_comm_rank = static_cast<INT>(comm_rank);

  auto loop = particle_loop(
      "DepartingIdentify", this->mpi_rank_dat,
      [=](auto loop_index, auto neso_mpi_rank) {
        const int cellx = loop_index.cell;
        const int layerx = loop_index.layer;
        const INT owning_rank = neso_mpi_rank[rank_component];

        // if rank is valid and not equal to this rank then this
        // particle is being sent somewhere
        if (((owning_rank >= 0) && (owning_rank < INT_comm_size)) &&
            (owning_rank != INT_comm_rank)) {
          // Increment the counter for the remote rank
          // reuse the recv ranks array to avoid mallocing more space
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              pack_layer_atomic(k_send_counts_all_ranks[owning_rank]);
          const int pack_layer = pack_layer_atomic.fetch_add(1);

          // increment the counter for number of sent particles (all
          // ranks)
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              send_count_atomic(k_num_particle_send[0]);
          const int send_index = send_count_atomic.fetch_add(1);

          // store the cell, source layer, packing layer
          k_pack_cells[send_index] = static_cast<int>(cellx);
          k_pack_layers_src[send_index] = static_cast<int>(layerx);
          k_pack_layers_dst[send_index] = pack_layer;

          // if the packing layer is zero then this is the first
          // particle found sending to the remote rank -> increment
          // the number of remote ranks and record this rank.
          if ((pack_layer == 0) && (rank_component == 0)) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                num_ranks_send_atomic(k_num_ranks_send[0]);
            const int rank_index = num_ranks_send_atomic.fetch_add(1);
            k_send_ranks[rank_index] = static_cast<int>(owning_rank);
            k_send_rank_map[owning_rank] = rank_index;
          }
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(this->mpi_rank_dat));

  loop->execute();

  auto e0 = this->dh_send_ranks.async_device_to_host();
  auto e1 = this->dh_send_counts_all_ranks.async_device_to_host();
  auto e2 = this->dh_send_rank_map.async_device_to_host();
  auto e3 = this->dh_num_ranks_send.async_device_to_host();
  auto e4 = this->dh_num_particle_send.async_device_to_host();

  e0.wait();
  e1.wait();
  e2.wait();
  e3.wait();
  e4.wait();

  sycl_target->profile_map.inc("DepartingIdentify", "identify", 1,
                               profile_elapsed(t0, profile_timestamp()));
}

} // namespace NESO::Particles

#endif
