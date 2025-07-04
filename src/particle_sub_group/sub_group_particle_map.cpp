#include <neso_particles/particle_sub_group/sub_group_particle_map.hpp>

namespace NESO::Particles {

std::tuple<int *, int *, INT *, INT *> SubGroupParticleMap::get_helper_ptrs() {
  return {this->dh_npart_cell->h_buffer.ptr, this->dh_npart_cell->d_buffer.ptr,
          this->dh_npart_cell_es->h_buffer.ptr,
          this->dh_npart_cell_es->d_buffer.ptr};
}

SubGroupParticleMap::SubGroupParticleMap(SYCLTargetSharedPtr sycl_target,
                                         const int cell_count)
    : sycl_target(sycl_target), cell_count(cell_count), cell_start(-1),
      cell_end(-1) {
  this->h_cell_starts =
      std::make_shared<BufferHost<INT *>>(sycl_target, cell_count);
  this->d_cell_starts =
      std::make_shared<BufferDevice<INT *>>(sycl_target, cell_count);
  this->d_layer_map =
      std::make_shared<BufferDevice<INT>>(sycl_target, cell_count);
  this->dh_npart_cell = std::make_shared<BufferDeviceHost<int>>(
      sycl_target, cell_count + cell_count * NESO_PARTICLES_CACHELINE_NUM_int);
  this->dh_npart_cell_es =
      std::make_shared<BufferDeviceHost<INT>>(sycl_target, cell_count);
}

void SubGroupParticleMap::create(const int cell_start, const int cell_end,
                                 const int *RESTRICT const h_cell_counts,
                                 const INT *RESTRICT const h_cell_counts_es) {
  NESOASSERT(cell_start >= 0 && cell_start <= cell_count, "Bad cell_start");
  NESOASSERT(cell_end >= 0 && cell_end <= cell_count, "Bad cell_end");
  this->cell_start = cell_start;
  this->cell_end = cell_end;
  if (cell_start < cell_end) {
    // Make sure the buffer is large enough to store the map.
    this->npart_total = h_cell_counts_es[cell_end - 1] +
                        static_cast<INT>(h_cell_counts[cell_end - 1]);
    this->d_layer_map->realloc_no_copy(
        static_cast<std::size_t>(this->npart_total));
    INT *ptr = this->d_layer_map->ptr;
    INT **h_cell_ptr = this->h_cell_starts->ptr;
    INT **d_cell_ptr = this->d_cell_starts->ptr;
    const std::size_t cell_count_range =
        static_cast<std::size_t>(cell_end - cell_start);
    for (int cx = cell_start; cx < cell_end; cx++) {
      h_cell_ptr[cx] = ptr + h_cell_counts_es[cx];
    }
    if (cell_count_range > 0) {
      this->sycl_target->queue
          .memcpy(d_cell_ptr + cell_start, h_cell_ptr + cell_start,
                  cell_count_range * sizeof(INT *))
          .wait_and_throw();
    }
  }
}

void SubGroupParticleMap::reset() {
  this->cell_start = this->cell_count + 1;
  this->cell_end = -1;
}

} // namespace NESO::Particles
