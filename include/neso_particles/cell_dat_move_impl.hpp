#ifndef _NESO_PARTICLES_CELL_DAT_MOVE_IMPL_H_
#define _NESO_PARTICLES_CELL_DAT_MOVE_IMPL_H_

#include "cell_dat_move.hpp"
#include "loop/particle_loop.hpp"

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

  if (this->ep_bad_cell_indices.get_flag()) {
    for (int cx = 0; cx < k_ncell; cx++) {
      auto CELLS = this->cell_id_dat->cell_dat.get_cell(cx);
      const int nrow = CELLS->nrow;
      for (int rx = 0; rx < nrow; rx++) {
        const int cell_on_dat = CELLS->at(rx, 0);
        const bool valid_cell = (cell_on_dat >= 0) && (cell_on_dat < k_ncell);
        if (!valid_cell) {
          this->print_particle(cx, rx);
        }
      }
    }
    NESOASSERT(false, "Particle held bad cell id (not in [0,..,N_cell - 1]). "
                      "Note N_cell is " +
                          std::to_string(k_ncell) + ".");
  }

  // Realloc the ParticleDat cells for the move
  if (this->ncell > 0) {
    this->sycl_target->queue
        .memcpy(this->h_npart_cell.ptr, this->d_npart_cell.ptr,
                sizeof(int) * this->ncell)
        .wait();
  }

  auto t0_realloc = profile_timestamp();
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
  sycl_target->profile_map.inc(
      "CellMove", "realloc", 1,
      profile_elapsed(t0_realloc, profile_timestamp()));

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

  this->sycl_target->profile_map.inc(
    "CellMove", "move_time", move_count, profile_elapsed(t1, profile_timestamp()));
  this->sycl_target->profile_map.set("CellMove", "num_bytes_per_particle", this->num_bytes_per_particle);

  auto t2 = profile_timestamp();

  // compress the data by removing the old rows
  this->layer_compressor.remove_particles(move_count, this->d_cells_old.ptr,
                                          this->d_layers_old.ptr);

  sycl_target->profile_map.inc("CellMove", "remove_particles", 1,
                               profile_elapsed(t2, profile_timestamp()));
  sycl_target->profile_map.inc("CellMove", "move", 1,
                               profile_elapsed(t0, profile_timestamp()));
}

// TODO make the actual call
inline void CellMove::move_test() {

  auto r0 = ProfileRegion("CellMove", "prepare");
  auto mpi_rank_dat = this->particle_dats_int[Sym<INT>("NESO_MPI_RANK")];
  const INT npart_local = mpi_rank_dat->get_npart_local();
  const auto k_ncell = this->ncell;

  auto k_npart_cell = this->d_npart_cell.ptr;
  auto k_npart_diff_cell = this->d_npart_diff_cell.ptr;
  auto k_mpi_npart_cell = mpi_rank_dat->d_npart_cell;

  this->d_cells_old.realloc_no_copy(npart_local);
  auto e_mask = this->sycl_target->queue.fill(this->d_cells_old.ptr, (int)-1,
                                              npart_local);

  this->d_cells_new.realloc_no_copy(npart_local);
  this->d_layers_old.realloc_no_copy(npart_local);
  this->d_layers_new.realloc_no_copy(npart_local);
  auto k_cells_old = this->d_cells_old.ptr;
  auto k_cells_new = this->d_cells_new.ptr;
  auto k_layers_old = this->d_layers_old.ptr;
  auto k_layers_new = this->d_layers_new.ptr;

  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(k_ncell), [=](sycl::id<1> idx) {
          k_npart_cell[idx] = k_mpi_npart_cell[idx];
        });
      })
      .wait_and_throw();
  e_mask.wait_and_throw();

  r0.end();
  this->sycl_target->profile_map.add_region(r0);

  // detect out of bounds particles
  auto k_ep_indices = this->ep_bad_cell_indices.device_ptr();

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

          const auto array_index = loop_index.get_loop_linear_index();

          k_cells_old[array_index] = cellx;
          k_cells_new[array_index] = static_cast<int>(cell_on_dat);
          k_layers_old[array_index] = layerx;
          k_layers_new[array_index] = layer_new;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(this->cell_id_dat))
      .execute();

  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(k_ncell), [=](sycl::id<1> idx) {
          k_npart_diff_cell[idx] = k_npart_cell[idx] - k_mpi_npart_cell[idx];
        });
      })
      .wait_and_throw();

  auto k_npart_exscan = this->dh_npart_exscan.d_buffer.ptr;
  auto event_exsum = joint_exclusive_scan(this->sycl_target, k_ncell + 1,
                                          k_npart_diff_cell, k_npart_exscan);

  if (this->ep_bad_cell_indices.get_flag()) {
    for (int cx = 0; cx < k_ncell; cx++) {
      auto CELLS = this->cell_id_dat->cell_dat.get_cell(cx);
      const int nrow = CELLS->nrow;
      for (int rx = 0; rx < nrow; rx++) {
        const int cell_on_dat = CELLS->at(rx, 0);
        const bool valid_cell = (cell_on_dat >= 0) && (cell_on_dat < k_ncell);
        if (!valid_cell) {
          this->print_particle(cx, rx);
        }
      }
    }
    NESOASSERT(false, "Particle held bad cell id (not in [0,..,N_cell - 1]). "
                      "Note N_cell is " +
                          std::to_string(k_ncell) + ".");
  }

  // Realloc the ParticleDat cells for the move
  if (this->ncell > 0) {
    this->sycl_target->queue
        .memcpy(this->h_npart_cell.ptr, this->d_npart_cell.ptr,
                sizeof(int) * this->ncell)
        .wait_and_throw();
  }

  auto r1 = ProfileRegion("CellMove", "realloc");
  auto t0_realloc = profile_timestamp();
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

  EventStack tmp_stack;
  for (auto &dat : particle_dats_real) {
    tmp_stack.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  for (auto &dat : particle_dats_int) {
    tmp_stack.push(dat.second->async_set_npart_cells(this->h_npart_cell));
  }
  tmp_stack.wait();

  r1.end();
  this->sycl_target->profile_map.add_region(r1);

  auto r2 = ProfileRegion("CellMove", "index_bookkeeping");
  event_exsum.wait_and_throw();

  this->dh_npart_exscan.device_to_host();
  const int total_moving_particles =
      this->dh_npart_exscan.h_buffer.ptr[k_ncell];

  this->d_ordered_cells_old.realloc_no_copy(total_moving_particles);
  this->d_ordered_cells_new.realloc_no_copy(total_moving_particles);
  this->d_ordered_layers_old.realloc_no_copy(total_moving_particles);
  this->d_ordered_layers_new.realloc_no_copy(total_moving_particles);
  auto k_ordered_cells_old = this->d_ordered_cells_old.ptr;
  auto k_ordered_cells_new = this->d_ordered_cells_new.ptr;
  auto k_ordered_layers_old = this->d_ordered_layers_old.ptr;
  auto k_ordered_layers_new = this->d_ordered_layers_new.ptr;

  tmp_stack.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<1>(npart_local), [=](sycl::id<1> idx) {
      const auto cell_old = k_cells_old[idx];
      if (cell_old > -1) {
        const auto cell_new = k_cells_new[idx];
        const auto layer_old = k_layers_old[idx];
        const auto layer_new = k_layers_new[idx];
        const int offset =
            layer_new - (k_npart_cell[cell_new] - k_npart_diff_cell[cell_new]);
        const auto index_base = k_npart_exscan[cell_new];
        const auto index = index_base + offset;
        k_ordered_cells_old[index] = cell_old;
        k_ordered_layers_old[index] = layer_old;
        k_ordered_layers_new[index] = layer_new;
      }
    });
  }));

  for (int cellx = 0; cellx < k_ncell; cellx++) {
    const std::size_t offset = this->dh_npart_exscan.h_buffer.ptr[cellx];
    const std::size_t count =
        this->dh_npart_exscan.h_buffer.ptr[cellx + 1] - offset;
    tmp_stack.push(this->sycl_target->queue.fill(k_ordered_cells_new + offset,
                                                 (int)cellx, count));
  }

  // Wait for all the fills and parallel_for we pushed onto the queue.
  tmp_stack.wait();
  r2.end();
  this->sycl_target->profile_map.add_region(r2);

  // we can now actually do the copy
  auto r3 = ProfileRegion("CellMove", "move");
  // get the pointers into the ParticleDats
  this->get_particle_dat_info();
  const int k_num_dats_real = this->num_dats_real;
  const int k_num_dats_int = this->num_dats_int;
  const auto k_particle_dat_ptr_real = this->d_particle_dat_ptr_real.ptr;
  const auto k_particle_dat_ptr_int = this->d_particle_dat_ptr_int.ptr;
  const auto k_particle_dat_ncomp_real = this->d_particle_dat_ncomp_real.ptr;
  const auto k_particle_dat_ncomp_int = this->d_particle_dat_ncomp_int.ptr;

  auto t0 = profile_timestamp();
/*
  tmp_stack.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<2>(k_num_dats_real, total_moving_particles),
                       [=](sycl::id<2> idx) {
                         const auto dx = idx[0];
                         const auto index = idx[1];

                         const auto cell_old = k_ordered_cells_old[index];
                         const auto cell_new = k_ordered_cells_new[index];
                         const auto layer_old = k_ordered_layers_old[index];
                         const auto layer_new = k_ordered_layers_new[index];

                         // loop over the ParticleDats and copy the data
                         // for each real dat
                         REAL ***dat_ptr = k_particle_dat_ptr_real[dx];
                         const int ncomp = k_particle_dat_ncomp_real[dx];
                         // for each component
                         for (int cx = 0; cx < ncomp; cx++) {
                           dat_ptr[cell_new][cx][layer_new] =
                               dat_ptr[cell_old][cx][layer_old];
                         }
                       });
  }));

  tmp_stack.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<2>(k_num_dats_int, total_moving_particles),
                       [=](sycl::id<2> idx) {
                         const auto dx = idx[0];
                         const auto index = idx[1];

                         const auto cell_old = k_ordered_cells_old[index];
                         const auto cell_new = k_ordered_cells_new[index];
                         const auto layer_old = k_ordered_layers_old[index];
                         const auto layer_new = k_ordered_layers_new[index];

                         // loop over the ParticleDats and copy the data
                         // for each int dat
                         INT ***dat_ptr = k_particle_dat_ptr_int[dx];
                         const int ncomp = k_particle_dat_ncomp_int[dx];
                         // for each component
                         for (int cx = 0; cx < ncomp; cx++) {
                           dat_ptr[cell_new][cx][layer_new] =
                               dat_ptr[cell_old][cx][layer_old];
                         }
                       });
  }));
*/

  tmp_stack.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<1>(total_moving_particles),
                       [=](sycl::id<1> idx) {
                         const auto index = idx;

                         const auto cell_old = k_ordered_cells_old[index];
                         const auto cell_new = k_ordered_cells_new[index];
                         const auto layer_old = k_ordered_layers_old[index];
                         const auto layer_new = k_ordered_layers_new[index];

                         for(int dx=0 ; dx<k_num_dats_real ; dx++){
                         // loop over the ParticleDats and copy the data
                         // for each real dat
                         REAL ***dat_ptr = k_particle_dat_ptr_real[dx];
                         const int ncomp = k_particle_dat_ncomp_real[dx];
                         // for each component
                         for (int cx = 0; cx < ncomp; cx++) {
                           dat_ptr[cell_new][cx][layer_new] =
                               dat_ptr[cell_old][cx][layer_old];
                         }
                         }
                       });
  }));

  tmp_stack.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(sycl::range<1>(total_moving_particles),
                       [=](sycl::id<1> idx) {
                         const auto index = idx;

                         const auto cell_old = k_ordered_cells_old[index];
                         const auto cell_new = k_ordered_cells_new[index];
                         const auto layer_old = k_ordered_layers_old[index];
                         const auto layer_new = k_ordered_layers_new[index];
                        for(int dx=0 ; dx<k_num_dats_int ; dx++){
                         // loop over the ParticleDats and copy the data
                         // for each int dat
                         INT ***dat_ptr = k_particle_dat_ptr_int[dx];
                         const int ncomp = k_particle_dat_ncomp_int[dx];
                         // for each component
                         for (int cx = 0; cx < ncomp; cx++) {
                           dat_ptr[cell_new][cx][layer_new] =
                               dat_ptr[cell_old][cx][layer_old];
                         }
                         }
                       });
  }));


  tmp_stack.wait();
  auto t1 = profile_timestamp();
  
  this->sycl_target->profile_map.inc(
    "CellMove", "move_time", total_moving_particles, profile_elapsed(t0, t1));
  this->sycl_target->profile_map.set("CellMove", "num_bytes_per_particle", this->num_bytes_per_particle);

  r3.end();
  this->sycl_target->profile_map.add_region(r3);

  // compress the data by removing the old rows
  auto r4 = ProfileRegion("CellMove", "layer_compress");
  this->layer_compressor.remove_particles(
      total_moving_particles, k_ordered_cells_old, k_ordered_layers_old);
  r4.end();
  this->sycl_target->profile_map.add_region(r4);
}

} // namespace NESO::Particles

#endif
