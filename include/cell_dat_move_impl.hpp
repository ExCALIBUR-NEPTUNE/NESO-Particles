#ifndef _NESO_PARTICLES_CELL_DAT_MOVE_IMPL_H_
#define _NESO_PARTICLES_CELL_DAT_MOVE_IMPL_H_

#include "cell_dat_move.hpp"
#include "loop/particle_loop.hpp"

using namespace cl;
namespace NESO::Particles {

/**
 * Move particles between cells (on this MPI rank) using the cell ids on
 * the particles.
 */
inline void CellMove::move() {
  auto r = ProfileRegion("CellMove", "prepare");
  auto t0 = profile_timestamp();
  // reset the particle counters on each cell
  auto mpi_rank_dat = particle_dats_int[Sym<INT>("NESO_MPI_RANK")];
  const auto k_ncell = this->ncell;
  auto k_npart_cell = d_npart_cell.ptr;
  auto k_mpi_npart_cell = mpi_rank_dat->d_npart_cell;
  auto reset_event = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<1>(k_ncell), [=](sycl::id<1> idx) {
      k_npart_cell[idx] = k_mpi_npart_cell[idx];
    });
  });
  auto k_move_count = d_move_count.ptr;
  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.single_task<>([=]() { k_move_count[0] = 0; });
      })
      .wait_and_throw();
  reset_event.wait_and_throw();

  // space to store particles moving between cells
  const INT npart_local = mpi_rank_dat->get_npart_local();
  this->d_cells_old.realloc_no_copy(npart_local);
  this->d_cells_new.realloc_no_copy(npart_local);
  this->d_layers_old.realloc_no_copy(npart_local);
  this->d_layers_new.realloc_no_copy(npart_local);

  auto k_cells_old = this->d_cells_old.ptr;
  auto k_cells_new = this->d_cells_new.ptr;
  auto k_layers_old = this->d_layers_old.ptr;
  auto k_layers_new = this->d_layers_new.ptr;

  // detect out of bounds particles
  auto k_ep_indices = this->ep_bad_cell_indices.device_ptr();

  r.end();
  this->sycl_target->profile_map.add_region(r);
  r = ProfileRegion("CellMove", "cell_move_identify");

  // loop over particles and identify the particles to be move between
  // cells.
  ParticleLoop(
      "cell_move_identify", this->cell_id_dat,
      [=](auto loop_index, auto cell_id_dat) {
        const INT cellx = loop_index.cell;
        const INT layerx = loop_index.layer;
        // if the cell on the particle is not the current cell then
        // the particle needs moving.
        const auto cell_on_dat = cell_id_dat[0];

        const bool valid_cell = (cell_on_dat >= 0) && (cell_on_dat < k_ncell);
        NESO_KERNEL_ASSERT(valid_cell, k_ep_indices);

        if ((cellx != cell_on_dat) && valid_cell) {
          // Atomically increment the particle count for the new
          // cell
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              atomic_layer(k_npart_cell[cell_on_dat]);
          const int layer_new = atomic_layer.fetch_add(1);

          // Get an index for this particle in the arrays that hold
          // old/new cells/layers
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              atomic_index(k_move_count[0]);
          const int array_index = atomic_index.fetch_add(1);

          k_cells_old[array_index] = cellx;
          k_cells_new[array_index] = static_cast<int>(cell_on_dat);
          k_layers_old[array_index] = layerx;
          k_layers_new[array_index] = layer_new;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(this->cell_id_dat))
      .execute();

  r.end();
  this->sycl_target->profile_map.add_region(r);
  r = ProfileRegion("CellMove", "realloc");

  this->ep_bad_cell_indices.check_and_throw(
      "Particle held bad cell id (not in [0,..,N_cell - 1]).");

  // Realloc the ParticleDat cells for the move
  if (this->ncell > 0) {
    this->sycl_target->queue
        .memcpy(this->h_npart_cell.ptr, this->d_npart_cell.ptr,
                sizeof(int) * this->ncell)
        .wait();
  }

  for (auto &dat : particle_dats_real) {
    dat.second->realloc(this->h_npart_cell);
  }
  for (auto &dat : particle_dats_int) {
    dat.second->realloc(this->h_npart_cell);
  }

  // wait for the reallocs
  for (auto &dat : particle_dats_real) {
    dat.second->wait_realloc();
  }
  for (auto &dat : particle_dats_int) {
    dat.second->wait_realloc();
  }
  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class dummy>(
            sycl::range<1>(k_ncell),
            [=](sycl::id<1> idx) { k_npart_cell[idx] = 0; });
      })
      .wait();

  EventStack tmp_stack;
  for (auto &dat : particle_dats_real) {
    tmp_stack.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  for (auto &dat : particle_dats_int) {
    tmp_stack.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  tmp_stack.wait();

  // get the npart to move on the host
  this->sycl_target->queue
      .memcpy(this->h_move_count.ptr, this->d_move_count.ptr, sizeof(int))
      .wait();
  const int move_count = h_move_count.ptr[0];

  r.end();
  this->sycl_target->profile_map.add_region(r);
  r = ProfileRegion("CellMove", "move");

  // get the pointers into the ParticleDats
  this->get_particle_dat_info();

  const int k_num_dats_real = this->num_dats_real;
  const int k_num_dats_int = this->num_dats_int;
  const auto k_particle_dat_ptr_real = this->d_particle_dat_ptr_real.ptr;
  const auto k_particle_dat_ptr_int = this->d_particle_dat_ptr_int.ptr;
  const auto k_particle_dat_ncomp_real = this->d_particle_dat_ncomp_real.ptr;
  const auto k_particle_dat_ncomp_int = this->d_particle_dat_ncomp_int.ptr;

  auto t1 = profile_timestamp();
  // copy from old cells/layers to new cells/layers
  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(move_count), [=](sycl::id<1> idx) {
          const auto cell_old = k_cells_old[idx];
          const auto cell_new = k_cells_new[idx];
          const auto layer_old = k_layers_old[idx];
          const auto layer_new = k_layers_new[idx];

          // loop over the ParticleDats and copy the data
          // for each real dat
          for (int dx = 0; dx < k_num_dats_real; dx++) {
            REAL ***dat_ptr = k_particle_dat_ptr_real[dx];
            const int ncomp = k_particle_dat_ncomp_real[dx];
            // for each component
            for (int cx = 0; cx < ncomp; cx++) {
              dat_ptr[cell_new][cx][layer_new] =
                  dat_ptr[cell_old][cx][layer_old];
            }
          }
          // for each int dat
          for (int dx = 0; dx < k_num_dats_int; dx++) {
            INT ***dat_ptr = k_particle_dat_ptr_int[dx];
            const int ncomp = k_particle_dat_ncomp_int[dx];
            // for each component
            for (int cx = 0; cx < ncomp; cx++) {
              dat_ptr[cell_new][cx][layer_new] =
                  dat_ptr[cell_old][cx][layer_old];
            }
          }
        });
      })
      .wait_and_throw();

  r.end();
  this->sycl_target->profile_map.add_region(r);

  sycl_target->profile_map.inc("CellMove", "cell_move", 1,
                               profile_elapsed(t1, profile_timestamp()));

  auto t2 = profile_timestamp();

  // compress the data by removing the old rows
  this->layer_compressor.remove_particles(move_count, this->d_cells_old.ptr,
                                          this->d_layers_old.ptr);

  sycl_target->profile_map.inc("CellMove", "remove_particles", 1,
                               profile_elapsed(t2, profile_timestamp()));
  sycl_target->profile_map.inc("CellMove", "move", 1,
                               profile_elapsed(t0, profile_timestamp()));
}

} // namespace NESO::Particles

#endif
