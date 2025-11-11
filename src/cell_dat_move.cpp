#include <neso_particles/cell_dat_move.hpp>
#include <neso_particles/common_impl.hpp>

namespace NESO::Particles {

void CellMove::move() {
  auto r = ProfileRegion("CellMove", "prepare");
  // reset the particle counters on each cell
  auto mpi_rank_dat = this->particle_group_pointer_map->particle_dats_int->at(
      Sym<INT>("NESO_MPI_RANK"));
  const auto k_ncell = this->particle_group_pointer_map->get_cell_count();

  auto d_npart_cell =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);

  const std::size_t cacheline_pad_int =
      this->sycl_target->device_limits.get_cacheline_size(sizeof(int));
  d_npart_cell->realloc_no_copy(k_ncell + k_ncell * cacheline_pad_int);
  auto k_npart_cell_seq = d_npart_cell->ptr;
  auto k_npart_cell = k_npart_cell_seq + k_ncell;

  auto k_mpi_npart_cell = mpi_rank_dat->d_npart_cell;
  auto reset_event = this->sycl_target->queue.parallel_for(
      sycl::range<1>(k_ncell),
      [=](sycl::id<1> idx) { k_npart_cell[idx * cacheline_pad_int] = 0; });

  auto d_cells_old =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  auto d_cells_new =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  auto d_layers_old =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  auto d_layers_new =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  auto h_npart_cell =
      get_resource<BufferHost<int>, ResourceStackInterfaceBufferHost<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferHost<int>{},
          sycl_target);
  h_npart_cell->realloc_no_copy(k_ncell);

  // space to store particles moving between cells
  const INT npart_local = mpi_rank_dat->get_npart_local();
  d_cells_old->realloc_no_copy(npart_local);
  d_cells_new->realloc_no_copy(npart_local);
  d_layers_old->realloc_no_copy(npart_local);
  d_layers_new->realloc_no_copy(npart_local);
  auto k_cells_old = d_cells_old->ptr;
  auto k_cells_new = d_cells_new->ptr;
  auto k_layers_old = d_layers_old->ptr;
  auto k_layers_new = d_layers_new->ptr;

  // detect out of bounds particles
  auto k_ep_indices = this->ep_bad_cell_indices.device_ptr();

  r.end();
  this->sycl_target->profile_map.add_region(r);
  r = ProfileRegion("CellMove", "cell_move_identify");

  auto d_tmp_cell_layers =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_tmp_cell_layers->realloc_no_copy(npart_local);
  auto k_tmp_cell_layers = d_tmp_cell_layers->ptr;

  reset_event.wait_and_throw();

  // loop over particles and identify the particles to be moved between
  // cells.
  ParticleLoop(
      "cell_move_identify", this->cell_id_dat,
      [=](auto loop_index, auto cell_id_dat) {
        const INT cellx = loop_index.cell;
        // if the cell on the particle is not the current cell then
        // the particle needs moving.
        const auto cell_on_dat = cell_id_dat.at(0);

        const bool valid_cell = (cell_on_dat >= 0) && (cell_on_dat < k_ncell);
        NESO_KERNEL_ASSERT(valid_cell, k_ep_indices);

        if ((cellx != cell_on_dat) && valid_cell) {
          // Atomically increment the offset count for the new
          // cell
          const int offset_new = atomic_fetch_add(
              &k_npart_cell[cell_on_dat * cacheline_pad_int], 1);

          k_tmp_cell_layers[loop_index.get_loop_linear_index()] = offset_new;
        } else {
          k_tmp_cell_layers[loop_index.get_loop_linear_index()] = -1;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(this->cell_id_dat))
      .execute();

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

  auto d_tmp_cell_new_npart =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_tmp_cell_new_npart->realloc_no_copy((k_ncell + 1) * 2);
  auto k_tmp_cell_new_npart = d_tmp_cell_new_npart->ptr;
  auto k_npart_cell_exscan = k_tmp_cell_new_npart + k_ncell + 1;

  auto e0 = this->sycl_target->queue.parallel_for(
      sycl::range<1>(k_ncell), [=](sycl::id<1> idx) {
        const int offset_total = k_npart_cell[idx * cacheline_pad_int];
        const int new_npart = offset_total + k_mpi_npart_cell[idx];
        k_npart_cell_seq[idx] = new_npart;
        k_tmp_cell_new_npart[idx] = offset_total;
      });

  auto e1 = this->sycl_target->queue.parallel_for(
      sycl::range<1>(k_ncell + 1),
      [=](sycl::id<1> idx) { k_npart_cell_exscan[idx] = 0; });
  e0.wait_and_throw();
  e1.wait_and_throw();

  auto event_h_npart_cell = this->sycl_target->queue.memcpy(
      h_npart_cell->ptr, k_npart_cell_seq, sizeof(int) * k_ncell);

  joint_exclusive_scan(this->sycl_target, static_cast<std::size_t>(k_ncell + 1),
                       k_tmp_cell_new_npart, k_npart_cell_exscan)
      .wait_and_throw();

  // get the npart to move on the host
  int move_count;
  auto event_move_count = this->sycl_target->queue.memcpy(
      &move_count, k_npart_cell_exscan + k_ncell, sizeof(int));

  event_move_count.wait_and_throw();
  NESOASSERT(move_count <= npart_local,
             "More particles are moving than exist locally.");

  ParticleLoop(
      "cell_move_identify_collect", this->cell_id_dat,
      [=](auto loop_index, auto cell_id_dat) {
        if (k_tmp_cell_layers[loop_index.get_loop_linear_index()] > -1) {
          const int offset_new =
              k_tmp_cell_layers[loop_index.get_loop_linear_index()];
          const auto cell_on_dat = cell_id_dat.at(0);
          const INT cellx = loop_index.cell;
          const int tmp_linear_index_exscan = k_npart_cell_exscan[cell_on_dat];
          const int array_index = offset_new + tmp_linear_index_exscan;
          const INT layerx = loop_index.layer;
          k_cells_old[array_index] = cellx;
          k_cells_new[array_index] = static_cast<int>(cell_on_dat);
          k_layers_old[array_index] = layerx;
          k_layers_new[array_index] =
              offset_new + k_mpi_npart_cell[cell_on_dat];
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(this->cell_id_dat))
      .execute();

  event_h_npart_cell.wait_and_throw();
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_tmp_cell_new_npart);
  k_tmp_cell_new_npart = nullptr;
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_tmp_cell_layers);
  k_tmp_cell_layers = nullptr;

  r.end();
  this->sycl_target->profile_map.add_region(r);
  r = ProfileRegion("CellMove", "realloc");

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_npart_cell);
  k_npart_cell = nullptr;
  k_npart_cell_seq = nullptr;
  k_npart_cell_exscan = nullptr;

  // Realloc the ParticleDat cells for the move
  for (auto &dat : *this->particle_group_pointer_map->particle_dats_real) {
    dat.second->realloc(*h_npart_cell);
  }
  for (auto &dat : *this->particle_group_pointer_map->particle_dats_int) {
    dat.second->realloc(*h_npart_cell);
  }

  // wait for the reallocs
  for (auto &dat : *this->particle_group_pointer_map->particle_dats_real) {
    dat.second->wait_realloc();
  }
  for (auto &dat : *this->particle_group_pointer_map->particle_dats_int) {
    dat.second->wait_realloc();
  }

  r.end();
  this->sycl_target->profile_map.add_region(r);
  r = ProfileRegion("CellMove", "move");

  // copy from old cells/layers to new cells/layers
  auto &device_ptr_map = this->particle_group_pointer_map->get();
  const int ncomp_total_real = device_ptr_map.ncomp_total_real;
  const int ncomp_total_int = device_ptr_map.ncomp_total_int;

  const int *RESTRICT k_flattened_dat_index_real =
      device_ptr_map.d_flattened_dat_index_real;
  const int *RESTRICT k_flattened_dat_index_int =
      device_ptr_map.d_flattened_dat_index_int;
  const int *RESTRICT k_flattened_comp_index_real =
      device_ptr_map.d_flattened_comp_index_real;
  const int *RESTRICT k_flattened_comp_index_int =
      device_ptr_map.d_flattened_comp_index_int;

  auto k_ptr_real = device_ptr_map.d_ptr_real;
  auto k_ptr_int = device_ptr_map.d_ptr_int;

  EventStack event_stack;
  if ((ncomp_total_real > 0) && (move_count > 0)) {
    event_stack.push(this->sycl_target->queue.parallel_for(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(ncomp_total_real, move_count)),
        [=](sycl::item<2> idx) {
          const int dat = k_flattened_dat_index_real[idx.get_id(0)];
          const int component = k_flattened_comp_index_real[idx.get_id(0)];
          const int particle = idx.get_id(1);

          const auto cell_old = k_cells_old[particle];
          const auto cell_new = k_cells_new[particle];
          const auto layer_old = k_layers_old[particle];
          const auto layer_new = k_layers_new[particle];

          k_ptr_real[dat][cell_new][component][layer_new] =
              k_ptr_real[dat][cell_old][component][layer_old];
        }));
  }

  if ((ncomp_total_int > 0) && (move_count > 0)) {
    event_stack.push(this->sycl_target->queue.parallel_for(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(ncomp_total_int, move_count)),
        [=](sycl::item<2> idx) {
          const int dat = k_flattened_dat_index_int[idx.get_id(0)];
          const int component = k_flattened_comp_index_int[idx.get_id(0)];
          const int particle = idx.get_id(1);

          const auto cell_old = k_cells_old[particle];
          const auto cell_new = k_cells_new[particle];
          const auto layer_old = k_layers_old[particle];
          const auto layer_new = k_layers_new[particle];

          k_ptr_int[dat][cell_new][component][layer_new] =
              k_ptr_int[dat][cell_old][component][layer_old];
        }));
  }

  this->particle_group_pointer_map->set_npart_cells_host(h_npart_cell->ptr);

  r.end();
  event_stack.wait();

  r.num_bytes = move_count *
                this->particle_group_pointer_map->get_num_bytes_per_particle() *
                2;
  this->sycl_target->profile_map.add_region(r);

  // compress the data by removing the old rows
  this->layer_compressor.remove_particles(move_count, k_cells_old,
                                          k_layers_old);

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_cells_old);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_cells_new);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_layers_old);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_layers_new);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferHost<int>{}, h_npart_cell);
}

} // namespace NESO::Particles
