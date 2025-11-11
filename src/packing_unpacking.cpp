#include <neso_particles/packing_unpacking.hpp>

namespace NESO::Particles {

char **ParticlePacker::get_packed_pointers(const int num_remote_send_ranks,
                                           const int *h_send_rank_npart_ptr) {
  this->h_send_pointers.realloc_no_copy(num_remote_send_ranks);
  NESOASSERT((this->cell_dat.ncells) >=
                 (this->sycl_target->comm_pair.size_parent),
             "Insuffient cells");

  if (this->device_aware_mpi_enabled) {
    for (int rankx = 0; rankx < num_remote_send_ranks; rankx++) {
      auto device_ptr = this->cell_dat.col_device_ptr(rankx, 0);
      this->h_send_pointers.ptr[rankx] = device_ptr;
    }
  } else {
    this->h_send_buffer.realloc_no_copy(this->required_send_buffer_length);
    INT offset = 0;
    std::stack<sycl::event> copy_events{};
    for (int rankx = 0; rankx < num_remote_send_ranks; rankx++) {
      const int npart_tmp = h_send_rank_npart_ptr[rankx];
      const int nbytes_tmp = npart_tmp * this->num_bytes_per_particle;

      auto device_ptr = this->cell_dat.col_device_ptr(rankx, 0);
      if (nbytes_tmp > 0) {
        copy_events.push(this->sycl_target->queue.memcpy(
            &this->h_send_buffer.ptr[offset], device_ptr, nbytes_tmp));
      }
      this->h_send_pointers.ptr[rankx] = &this->h_send_buffer.ptr[offset];
      offset += nbytes_tmp;
    }

    while (!copy_events.empty()) {
      auto event = copy_events.top();
      event.wait_and_throw();
      copy_events.pop();
    }
  }
  return this->h_send_pointers.ptr;
}

void ParticlePacker::pack(
    const int num_remote_send_ranks, BufferHost<int> &h_send_rank_npart,
    BufferDeviceHost<int> &dh_send_rank_map, const int num_particles_leaving,
    BufferDevice<int> &d_pack_cells, BufferDevice<int> &d_pack_layers_src,
    BufferDevice<int> &d_pack_layers_dst,
    ParticleGroupPointerMapSharedPtr particle_group_pointer_map,
    EventStack &event_stack, const int rank_component) {

  // Allocate enough space to store the particles to pack
  const std::size_t k_num_bytes_per_particle =
      particle_group_pointer_map->get_num_bytes_per_particle();
  this->num_bytes_per_particle = k_num_bytes_per_particle;

  const std::size_t k_num_real_bytes_per_particle =
      particle_group_pointer_map->get_num_real_bytes_per_particle();

  this->required_send_buffer_length = 0;
  for (int rankx = 0; rankx < num_remote_send_ranks; rankx++) {
    const int npart = h_send_rank_npart.ptr[rankx];
    const INT rankx_contrib = npart * k_num_bytes_per_particle;
    this->cell_dat.set_nrow(rankx, rankx_contrib);
    this->required_send_buffer_length += rankx_contrib;
  }
  this->cell_dat.wait_set_nrow();

  auto *particle_dats_int = particle_group_pointer_map->particle_dats_int;
  const auto k_particle_dat_rank =
      particle_dats_int->at(Sym<INT>("NESO_MPI_RANK"))->impl_get_const();
  const auto k_send_rank_map = dh_send_rank_map.d_buffer.ptr;
  const int k_rank_component = rank_component;

  const auto k_pack_cells = d_pack_cells.ptr;
  const auto k_pack_layers_src = d_pack_layers_src.ptr;
  const auto k_pack_layers_dst = d_pack_layers_dst.ptr;
  auto k_pack_cell_dat = this->cell_dat.device_ptr();

  auto &device_ptr_map = particle_group_pointer_map->get_const();
  const int ncomp_total_real = device_ptr_map.ncomp_total_real;
  const int ncomp_total_int = device_ptr_map.ncomp_total_int;

  const int *RESTRICT k_ncomp_exscan_real = device_ptr_map.d_ncomp_exscan_real;
  const int *RESTRICT k_ncomp_exscan_int = device_ptr_map.d_ncomp_exscan_int;

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

  if (num_particles_leaving <= 0) {
    return;
  }

  event_stack.push(this->sycl_target->queue.parallel_for<>(
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<2>(static_cast<std::size_t>(num_particles_leaving),
                         static_cast<std::size_t>(ncomp_total_real))),
      [=](sycl::item<2> idx) {
        const std::size_t particle = idx.get_id(0);
        const int dat = k_flattened_dat_index_real[idx.get_id(1)];
        const int dat_component = k_flattened_comp_index_real[idx.get_id(1)];
        const int index = k_ncomp_exscan_real[dat] + dat_component;

        const int cell = k_pack_cells[particle];
        const int layer_src = k_pack_layers_src[particle];
        const int layer_dst = k_pack_layers_dst[particle];
        const int rank = k_particle_dat_rank[cell][k_rank_component][layer_src];
        const int rank_packing_cell = k_send_rank_map[rank];

        char *base_pack_ptr =
            &k_pack_cell_dat[rank_packing_cell][0]
                            [layer_dst * k_num_bytes_per_particle];
        REAL *pack_ptr_real = (REAL *)base_pack_ptr;
        pack_ptr_real[index] = k_ptr_real[dat][cell][dat_component][layer_src];
      }));

  event_stack.push(this->sycl_target->queue.parallel_for<>(
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<2>(static_cast<std::size_t>(num_particles_leaving),
                         static_cast<std::size_t>(ncomp_total_int))),
      [=](sycl::item<2> idx) {
        const std::size_t particle = idx.get_id(0);
        const int dat = k_flattened_dat_index_int[idx.get_id(1)];
        const int dat_component = k_flattened_comp_index_int[idx.get_id(1)];
        const int index = k_ncomp_exscan_int[dat] + dat_component;

        const int cell = k_pack_cells[particle];
        const int layer_src = k_pack_layers_src[particle];
        const int layer_dst = k_pack_layers_dst[particle];
        const int rank = k_particle_dat_rank[cell][k_rank_component][layer_src];
        const int rank_packing_cell = k_send_rank_map[rank];

        char *base_pack_ptr =
            &k_pack_cell_dat[rank_packing_cell][0]
                            [layer_dst * k_num_bytes_per_particle];
        // offset for the REAL components
        base_pack_ptr += k_num_real_bytes_per_particle;

        INT *pack_ptr_int = (INT *)base_pack_ptr;
        pack_ptr_int[index] = k_ptr_int[dat][cell][dat_component][layer_src];
      }));
}

char **ParticleUnpacker::get_recv_pointers(const int num_remote_recv_ranks) {
  this->h_recv_pointers.realloc_no_copy(num_remote_recv_ranks);

  NESOASSERT(this->num_ranks_reset == num_remote_recv_ranks,
             "Missmatch in expected number of ranks. Was reset called?");

  if (this->device_aware_mpi_enabled) {
    for (int rankx = 0; rankx < num_remote_recv_ranks; rankx++) {
      const auto offset = this->h_recv_offsets.ptr[rankx];
      auto ptr = &this->d_recv_buffer.ptr[offset];
      this->h_recv_pointers.ptr[rankx] = ptr;
    }
  } else {
    for (int rankx = 0; rankx < num_remote_recv_ranks; rankx++) {
      const auto offset = this->h_recv_offsets.ptr[rankx];
      auto ptr = &this->h_recv_buffer.ptr[offset];
      this->h_recv_pointers.ptr[rankx] = ptr;
    }
  }

  return this->h_recv_pointers.ptr;
}

void ParticleUnpacker::reset(
    const int num_remote_recv_ranks, BufferHost<int> &h_recv_rank_npart,
    ParticleGroupPointerMapSharedPtr particle_group_pointer_map) {
  this->num_ranks_reset = num_remote_recv_ranks;

  // realloc the array that holds where in the recv buffer the data from each
  // remote rank should be placed
  this->h_recv_offsets.realloc_no_copy(num_remote_recv_ranks);
  this->num_bytes_per_particle =
      particle_group_pointer_map->get_num_bytes_per_particle();

  // compute the offsets in the recv buffer
  this->npart_recv = 0;
  for (int rankx = 0; rankx < num_remote_recv_ranks; rankx++) {
    this->h_recv_offsets.ptr[rankx] =
        this->npart_recv * this->num_bytes_per_particle;
    this->npart_recv += h_recv_rank_npart.ptr[rankx];
  }

  // realloc the recv buffer
  if (!this->device_aware_mpi_enabled) {
    this->h_recv_buffer.realloc_no_copy(this->npart_recv *
                                        this->num_bytes_per_particle);
  }
  this->d_recv_buffer.realloc_no_copy(this->npart_recv *
                                      this->num_bytes_per_particle);
}

void ParticleUnpacker::unpack(
    ParticleGroupPointerMapSharedPtr particle_group_pointer_map) {

  auto r0 = this->sycl_target->profile_map.start_region("ParticleUnpacker",
                                                        "unpack_realloc");
  // copy packed data to device
  const int cpysize = this->npart_recv * this->num_bytes_per_particle;
  sycl::event event_memcpy;
  if ((cpysize > 0) && (!this->device_aware_mpi_enabled)) {
    event_memcpy = this->sycl_target->queue.memcpy(
        this->d_recv_buffer.ptr, this->h_recv_buffer.ptr, cpysize);
  }

  auto &particle_dats_int = *particle_group_pointer_map->particle_dats_int;
  auto &particle_dats_real = *particle_group_pointer_map->particle_dats_real;

  // old cell occupancy
  auto mpi_rank_dat = particle_dats_int[Sym<INT>("NESO_MPI_RANK")];
  const int npart_cell_0_old = mpi_rank_dat->h_npart_cell[0];
  const int npart_cell_0_new = npart_cell_0_old + this->npart_recv;
  // realloc cell 0 on the dats
  for (auto &dat : particle_dats_real) {
    dat.second->realloc(0, npart_cell_0_new);
    dat.second->set_npart_cell(0, npart_cell_0_new);
  }
  for (auto &dat : particle_dats_int) {
    dat.second->realloc(0, npart_cell_0_new);
    dat.second->set_npart_cell(0, npart_cell_0_new);
  }
  for (auto &dat : particle_dats_real) {
    dat.second->wait_realloc();
  }
  for (auto &dat : particle_dats_int) {
    dat.second->wait_realloc();
  }

  this->sycl_target->profile_map.end_region(r0);
  const int k_npart_recv = this->npart_recv;

  auto &device_ptr_map = particle_group_pointer_map->get();
  const int ncomp_total_real = device_ptr_map.ncomp_total_real;
  const int ncomp_total_int = device_ptr_map.ncomp_total_int;
  const int k_num_bytes_per_particle = this->num_bytes_per_particle;
  char *k_recv_buffer = this->d_recv_buffer.ptr;
  const std::size_t k_num_real_bytes_per_particle =
      particle_group_pointer_map->get_num_real_bytes_per_particle();
  this->num_bytes_per_particle = k_num_bytes_per_particle;

  const int *RESTRICT k_ncomp_exscan_real = device_ptr_map.d_ncomp_exscan_real;
  const int *RESTRICT k_ncomp_exscan_int = device_ptr_map.d_ncomp_exscan_int;

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

  if ((cpysize > 0) && (!this->device_aware_mpi_enabled)) {
    event_memcpy.wait_and_throw();
  }

  if (k_npart_recv <= 0) {
    return;
  }
  auto r1 = this->sycl_target->profile_map.start_region("ParticleUnpacker",
                                                        "unpack_unpack");
  EventStack event_stack;
  event_stack.push(this->sycl_target->queue.parallel_for<>(
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<2>(static_cast<std::size_t>(k_npart_recv),
                         static_cast<std::size_t>(ncomp_total_real))),
      [=](sycl::item<2> idx) {
        const int particle = idx.get_id(0);
        const int dat = k_flattened_dat_index_real[idx.get_id(1)];
        const int dat_component = k_flattened_comp_index_real[idx.get_id(1)];
        const int index = k_ncomp_exscan_real[dat] + dat_component;

        const int layer = npart_cell_0_old + particle;
        const int offset = k_num_bytes_per_particle * particle;

        const char *const unpack_base_ptr = k_recv_buffer + offset;
        const REAL *const unpack_ptr_real = (REAL *const)unpack_base_ptr;
        k_ptr_real[dat][0][dat_component][layer] = unpack_ptr_real[index];
      }));

  event_stack.push(this->sycl_target->queue.parallel_for<>(
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<2>(static_cast<std::size_t>(k_npart_recv),
                         static_cast<std::size_t>(ncomp_total_int))),
      [=](sycl::item<2> idx) {
        const int particle = idx.get_id(0);
        const int dat = k_flattened_dat_index_int[idx.get_id(1)];
        const int dat_component = k_flattened_comp_index_int[idx.get_id(1)];
        const int index = k_ncomp_exscan_int[dat] + dat_component;

        const int layer = npart_cell_0_old + particle;
        const int offset = k_num_bytes_per_particle * particle;

        const char *const unpack_base_ptr =
            k_recv_buffer + offset + k_num_real_bytes_per_particle;
        const INT *const unpack_ptr_int = (INT *const)unpack_base_ptr;
        k_ptr_int[dat][0][dat_component][layer] = unpack_ptr_int[index];
      }));

  event_stack.wait();

  this->sycl_target->profile_map.end_region(r1);
}

} // namespace NESO::Particles
