#ifndef _NESO_PARTICLES_DEPARTING_PARTICLE_IDENTIFICATION
#define _NESO_PARTICLES_DEPARTING_PARTICLE_IDENTIFICATION

#include <CL/sycl.hpp>
#include <mpi.h>

#include "communication.hpp"
#include "compute_target.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

class DepartingIdentify {

private:
public:
  BufferDeviceHost<int> dh_send_ranks;
  BufferDeviceHost<int> dh_send_counts_all_ranks;
  BufferDeviceHost<int> dh_send_rank_map;
  BufferDevice<int> d_pack_cells;
  BufferDevice<int> d_pack_layers_src;
  BufferDevice<int> d_pack_layers_dst;
  BufferDeviceHost<int> dh_num_ranks_send;
  BufferDeviceHost<int> dh_num_particle_send;

  SYCLTarget &sycl_target;
  ParticleDatShPtr<INT> mpi_rank_dat;

  ~DepartingIdentify(){};
  DepartingIdentify(SYCLTarget &sycl_target)
      : sycl_target(sycl_target),
        dh_send_ranks(sycl_target, sycl_target.comm_pair.size_parent),
        dh_send_counts_all_ranks(sycl_target,
                                 sycl_target.comm_pair.size_parent),
        dh_send_rank_map(sycl_target, sycl_target.comm_pair.size_parent),
        d_pack_cells(sycl_target, 1), d_pack_layers_src(sycl_target, 1),
        d_pack_layers_dst(sycl_target, 1), dh_num_ranks_send(sycl_target, 1),
        dh_num_particle_send(sycl_target, 1){};

  inline void set_mpi_rank_dat(ParticleDatShPtr<INT> mpi_rank_dat) {
    this->mpi_rank_dat = mpi_rank_dat;
  }

  inline void identify(const int rank_component = 0) {
    auto t0 = profile_timestamp();

    const int comm_size = this->sycl_target.comm_pair.size_parent;
    const int comm_rank = this->sycl_target.comm_pair.rank_parent;

    auto pl_iter_range = this->mpi_rank_dat->get_particle_loop_iter_range();
    auto pl_stride = this->mpi_rank_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->mpi_rank_dat->get_particle_loop_npart_cell();

    const auto npart_upper_bound = this->mpi_rank_dat->get_npart_upper_bound();
    this->d_pack_cells.realloc_no_copy(npart_upper_bound);
    this->d_pack_layers_src.realloc_no_copy(npart_upper_bound);
    this->d_pack_layers_dst.realloc_no_copy(npart_upper_bound);

    auto k_send_ranks = this->dh_send_ranks.d_buffer.ptr;
    auto k_send_counts_all_ranks = this->dh_send_counts_all_ranks.d_buffer.ptr;
    auto k_num_ranks_send = this->dh_num_ranks_send.d_buffer.ptr;
    auto k_send_rank_map = this->dh_send_rank_map.d_buffer.ptr;
    auto k_pack_cells = this->d_pack_cells.ptr;
    auto k_pack_layers_src = this->d_pack_layers_src.ptr;
    auto k_pack_layers_dst = this->d_pack_layers_dst.ptr;
    auto k_num_particle_send = this->dh_num_particle_send.d_buffer.ptr;

    // zero the send/recv counts
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<1>(comm_size), [=](sycl::id<1> idx) {
        k_send_ranks[idx] = 0;
        k_send_counts_all_ranks[idx] = 0;
      });
    });
    // zero the number of ranks involved with send/recv
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<>([=]() {
        k_num_ranks_send[0] = 0;
        k_num_particle_send[0] = 0;
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
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                const INT owning_rank =
                    d_neso_mpi_rank[cellx][rank_component][layerx];

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
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

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

    sycl_target.profile_map.inc("DepartingIdentify", "identify", 1,
                                profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO::Particles

#endif
