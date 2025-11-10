#include <neso_particles/packing_unpacking.hpp>

namespace NESO::Particles {

size_t ParticlePacker::particle_size(
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int) {
  size_t s = 0;
  for (auto &dat : particle_dats_real) {
    s += dat.second->cell_dat.row_size();
  }
  for (auto &dat : particle_dats_int) {
    s += dat.second->cell_dat.row_size();
  }
  return s;
}

void ParticlePacker::get_particle_dat_info(
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int) {

  num_dats_real = particle_dats_real.size();
  dh_particle_dat_ptr_real.realloc_no_copy(num_dats_real);
  dh_particle_dat_ncomp_real.realloc_no_copy(num_dats_real);

  num_dats_int = particle_dats_int.size();
  dh_particle_dat_ptr_int.realloc_no_copy(num_dats_int);
  dh_particle_dat_ncomp_int.realloc_no_copy(num_dats_int);

  int index = 0;
  for (auto &dat : particle_dats_real) {
    dh_particle_dat_ptr_real.h_buffer.ptr[index] = dat.second->impl_get_const();
    dh_particle_dat_ncomp_real.h_buffer.ptr[index] = dat.second->ncomp;
    index++;
  }
  auto e0 = dh_particle_dat_ptr_real.async_host_to_device();
  auto e1 = dh_particle_dat_ncomp_real.async_host_to_device();
  index = 0;
  for (auto &dat : particle_dats_int) {
    dh_particle_dat_ptr_int.h_buffer.ptr[index] = dat.second->impl_get_const();
    dh_particle_dat_ncomp_int.h_buffer.ptr[index] = dat.second->ncomp;
    index++;
  }

  auto e2 = dh_particle_dat_ptr_int.async_host_to_device();
  auto e3 = dh_particle_dat_ncomp_int.async_host_to_device();

  e0.wait();
  e1.wait();
  e2.wait();
  e3.wait();
}

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

  auto &device_ptr_map = particle_group_pointer_map->get();
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

sycl::event ParticlePacker::pack(
    const int num_remote_send_ranks, BufferHost<int> &h_send_rank_npart,
    BufferDeviceHost<int> &dh_send_rank_map, const int num_particles_leaving,
    BufferDevice<int> &d_pack_cells, BufferDevice<int> &d_pack_layers_src,
    BufferDevice<int> &d_pack_layers_dst,
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int,
    const int rank_component) {

  auto t0 = profile_timestamp();

  // Allocate enough space to store the particles to pack
  this->num_bytes_per_particle =
      particle_size(particle_dats_real, particle_dats_int);

  this->required_send_buffer_length = 0;
  for (int rankx = 0; rankx < num_remote_send_ranks; rankx++) {
    const int npart = h_send_rank_npart.ptr[rankx];
    const INT rankx_contrib = npart * this->num_bytes_per_particle;
    this->cell_dat.set_nrow(rankx, rankx_contrib);
    this->required_send_buffer_length += rankx_contrib;
  }
  this->cell_dat.wait_set_nrow();

  // get the pointers to the particle dat data and the number of components in
  // each dat
  get_particle_dat_info(particle_dats_real, particle_dats_int);

  // loop over the particles to pack and for each particle pack the data into
  // the CellDat where each cell is the data to send to a remote rank.

  const int k_num_dats_real = this->num_dats_real;
  const int k_num_dats_int = this->num_dats_int;

  const auto k_particle_dat_ptr_real = dh_particle_dat_ptr_real.d_buffer.ptr;
  const auto k_particle_dat_ptr_int = dh_particle_dat_ptr_int.d_buffer.ptr;
  const auto k_particle_dat_ncomp_real =
      dh_particle_dat_ncomp_real.d_buffer.ptr;
  const auto k_particle_dat_ncomp_int = dh_particle_dat_ncomp_int.d_buffer.ptr;
  const auto k_particle_dat_rank =
      particle_dats_int[Sym<INT>("NESO_MPI_RANK")]->impl_get_const();
  const auto k_send_rank_map = dh_send_rank_map.d_buffer.ptr;
  const int k_rank_component = rank_component;

  const auto k_pack_cells = d_pack_cells.ptr;
  const auto k_pack_layers_src = d_pack_layers_src.ptr;
  const auto k_pack_layers_dst = d_pack_layers_dst.ptr;

  const int k_num_bytes_per_particle = this->num_bytes_per_particle;

  auto k_pack_cell_dat = this->cell_dat.device_ptr();

  sycl_target->profile_map.inc("ParticlePacker", "pack_prepare", 1,
                               profile_elapsed(t0, profile_timestamp()));

  sycl::event event;

  if (num_particles_leaving > 0) {
    event = this->sycl_target->queue.parallel_for<>(
        // for each leaving particle
        sycl::range<1>(static_cast<size_t>(num_particles_leaving)),
        [=](sycl::id<1> idx) {
          const int cell = k_pack_cells[idx];
          const int layer_src = k_pack_layers_src[idx];
          const int layer_dst = k_pack_layers_dst[idx];
          const int rank =
              k_particle_dat_rank[cell][k_rank_component][layer_src];
          const int rank_packing_cell = k_send_rank_map[rank];

          char *base_pack_ptr =
              &k_pack_cell_dat[rank_packing_cell][0]
                              [layer_dst * k_num_bytes_per_particle];
          REAL *pack_ptr_real = (REAL *)base_pack_ptr;
          // for each real dat
          int index = 0;
          for (int dx = 0; dx < k_num_dats_real; dx++) {
            auto dat_ptr = k_particle_dat_ptr_real[dx];
            const int ncomp = k_particle_dat_ncomp_real[dx];
            // for each component
            for (int cx = 0; cx < ncomp; cx++) {
              pack_ptr_real[index + cx] = dat_ptr[cell][cx][layer_src];
            }
            index += ncomp;
          }
          // for each int dat
          INT *pack_ptr_int = (INT *)(pack_ptr_real + index);
          index = 0;
          for (int dx = 0; dx < k_num_dats_int; dx++) {
            auto dat_ptr = k_particle_dat_ptr_int[dx];
            const int ncomp = k_particle_dat_ncomp_int[dx];
            // for each component
            for (int cx = 0; cx < ncomp; cx++) {
              pack_ptr_int[index + cx] = dat_ptr[cell][cx][layer_src];
            }
            index += ncomp;
          }
        });
  }

  return event;
}

size_t ParticleUnpacker::particle_size(
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int) {
  size_t s = 0;
  for (auto &dat : particle_dats_real) {
    s += dat.second->cell_dat.row_size();
  }
  for (auto &dat : particle_dats_int) {
    s += dat.second->cell_dat.row_size();
  }
  this->num_bytes_per_particle = s;
  return s;
}

void ParticleUnpacker::get_particle_dat_info(
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int) {

  auto r = ProfileRegion("ParticleUnpacker", "get_particle_dat_info");

  num_dats_real = particle_dats_real.size();
  dh_particle_dat_ptr_real.realloc_no_copy(num_dats_real);
  dh_particle_dat_ncomp_real.realloc_no_copy(num_dats_real);

  num_dats_int = particle_dats_int.size();
  dh_particle_dat_ptr_int.realloc_no_copy(num_dats_int);
  dh_particle_dat_ncomp_int.realloc_no_copy(num_dats_int);

  int index = 0;
  for (auto &dat : particle_dats_real) {
    dh_particle_dat_ptr_real.h_buffer.ptr[index] = dat.second->impl_get();
    dh_particle_dat_ncomp_real.h_buffer.ptr[index] = dat.second->ncomp;
    index++;
  }
  auto e0 = dh_particle_dat_ptr_real.async_host_to_device();
  auto e1 = dh_particle_dat_ncomp_real.async_host_to_device();
  index = 0;
  for (auto &dat : particle_dats_int) {
    dh_particle_dat_ptr_int.h_buffer.ptr[index] = dat.second->impl_get();
    dh_particle_dat_ncomp_int.h_buffer.ptr[index] = dat.second->ncomp;
    index++;
  }

  auto e2 = dh_particle_dat_ptr_int.async_host_to_device();
  auto e3 = dh_particle_dat_ncomp_int.async_host_to_device();

  e0.wait();
  e1.wait();
  e2.wait();
  e3.wait();

  r.end();
  this->sycl_target->profile_map.add_region(r);
}

char **ParticleUnpacker::get_recv_pointers(const int num_remote_recv_ranks) {
  this->h_recv_pointers.realloc_no_copy(num_remote_recv_ranks);

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
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int) {

  // realloc the array that holds where in the recv buffer the data from each
  // remote rank should be placed
  this->h_recv_offsets.realloc_no_copy(num_remote_recv_ranks);
  this->num_bytes_per_particle =
      this->particle_size(particle_dats_real, particle_dats_int);

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
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int) {

  auto t0 = profile_timestamp();
  auto r = ProfileRegion("unpack", "realloc");

  // copy packed data to device

  const int cpysize = this->npart_recv * this->num_bytes_per_particle;
  sycl::event event_memcpy;
  if ((cpysize > 0) && (!this->device_aware_mpi_enabled)) {
    event_memcpy = this->sycl_target->queue.memcpy(
        this->d_recv_buffer.ptr, this->h_recv_buffer.ptr, cpysize);
  }

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

  r.end();
  this->sycl_target->profile_map.add_region(r);

  const int k_npart_recv = this->npart_recv;

  // get the pointers to the particle dat data and the number of components in
  // each dat
  get_particle_dat_info(particle_dats_real, particle_dats_int);

  // unpack into cell 0
  const int k_num_bytes_per_particle = this->num_bytes_per_particle;
  const int k_num_dats_real = this->num_dats_real;
  const int k_num_dats_int = this->num_dats_int;
  const auto k_particle_dat_ptr_real =
      this->dh_particle_dat_ptr_real.d_buffer.ptr;
  const auto k_particle_dat_ptr_int =
      this->dh_particle_dat_ptr_int.d_buffer.ptr;
  const auto k_particle_dat_ncomp_real =
      this->dh_particle_dat_ncomp_real.d_buffer.ptr;
  const auto k_particle_dat_ncomp_int =
      this->dh_particle_dat_ncomp_int.d_buffer.ptr;
  char *k_recv_buffer = this->d_recv_buffer.ptr;

  sycl_target->profile_map.inc("ParticleUnpacker", "unpack_prepare", 1,
                               profile_elapsed(t0, profile_timestamp()));

  if ((cpysize > 0) && (!this->device_aware_mpi_enabled)) {
    event_memcpy.wait_and_throw();
  }

  r = ProfileRegion("unpack", "unpack_loop");

  if (k_npart_recv > 0) {
    this->sycl_target->queue
        .parallel_for(
            // for each new particle
            sycl::range<1>(static_cast<size_t>(k_npart_recv)),
            [=](sycl::id<1> idx) {
              const int cell = 0;
              // destination layer in the cell
              const int layer = npart_cell_0_old + idx;
              // source position in the packed buffer
              const int offset = k_num_bytes_per_particle * idx;
              char *unpack_base_ptr = k_recv_buffer + offset;
              REAL *unpack_ptr_real = (REAL *)unpack_base_ptr;
              // for each real dat
              int index = 0;
              for (int dx = 0; dx < k_num_dats_real; dx++) {
                REAL ***dat_ptr = k_particle_dat_ptr_real[dx];
                const int ncomp = k_particle_dat_ncomp_real[dx];
                // for each component
                for (int cx = 0; cx < ncomp; cx++) {
                  dat_ptr[cell][cx][layer] = unpack_ptr_real[index + cx];
                }
                index += ncomp;
              }
              // for each int dat
              INT *unpack_ptr_int = (INT *)(unpack_ptr_real + index);
              index = 0;
              for (int dx = 0; dx < k_num_dats_int; dx++) {
                INT ***dat_ptr = k_particle_dat_ptr_int[dx];
                const int ncomp = k_particle_dat_ncomp_int[dx];
                // for each component
                for (int cx = 0; cx < ncomp; cx++) {
                  dat_ptr[cell][cx][layer] = unpack_ptr_int[index + cx];
                }
                index += ncomp;
              }
            })
        .wait_and_throw();
  }
  r.end();
  this->sycl_target->profile_map.add_region(r);
}

} // namespace NESO::Particles
