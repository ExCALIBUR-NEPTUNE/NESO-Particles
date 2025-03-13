#ifndef _NESO_PARTICLES_SUB_GROUP_SUB_GROUP_PARTICLE_MAP_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SUB_GROUP_PARTICLE_MAP_HPP_

#include "../compute_target.hpp"
#include "../device_buffers.hpp"
#include "../typedefs.hpp"

namespace NESO::Particles {

/**
 * Container to store a device map from looping cell/layer to actual particle
 * cell/layer.
 */
struct SubGroupParticleMap {
  SYCLTargetSharedPtr sycl_target;
  // The number of cells in the map.
  int cell_count;
  // Holds pointers into d_layer_map.
  std::shared_ptr<BufferHost<INT *>> h_cell_starts;
  std::shared_ptr<BufferDevice<INT *>> d_cell_starts;
  // The actual map stored in a single array.
  std::shared_ptr<BufferDevice<INT>> d_layer_map;
  // The cell start and end that were passed to create.
  int cell_start, cell_end;

  // Helper buffers that the user will need to create the Selection and map.
  // These buffers will not be modified internally.
  std::shared_ptr<BufferDeviceHost<int>> dh_npart_cell;
  std::shared_ptr<BufferDeviceHost<INT>> dh_npart_cell_es;

  /**
   * @returns the host and device pointers for dh_npart_cell and
   * dh_npart_cell_es.
   */
  inline std::tuple<int *, int *, INT *, INT *> get_helper_ptrs() {
    return {this->dh_npart_cell->h_buffer.ptr,
            this->dh_npart_cell->d_buffer.ptr,
            this->dh_npart_cell_es->h_buffer.ptr,
            this->dh_npart_cell_es->d_buffer.ptr};
  }

  /**
   * @param cell_count The number of cells to create a map for.
   */
  SubGroupParticleMap(SYCLTargetSharedPtr sycl_target, const int cell_count)
      : sycl_target(sycl_target), cell_count(cell_count), cell_start(-1),
        cell_end(-1) {
    this->h_cell_starts =
        std::make_shared<BufferHost<INT *>>(sycl_target, cell_count);
    this->d_cell_starts =
        std::make_shared<BufferDevice<INT *>>(sycl_target, cell_count);
    this->d_layer_map =
        std::make_shared<BufferDevice<INT>>(sycl_target, cell_count);
    this->dh_npart_cell =
        std::make_shared<BufferDeviceHost<int>>(sycl_target, cell_count);
    this->dh_npart_cell_es =
        std::make_shared<BufferDeviceHost<INT>>(sycl_target, cell_count);
  }

  /**
   * Create a map for a range of cells.
   */
  inline void create(const int cell_start, const int cell_end,
                     const int *RESTRICT const h_cell_counts,
                     const INT *RESTRICT const h_cell_counts_es) {
    NESOASSERT(cell_start >= 0 && cell_start <= cell_count, "Bad cell_start");
    NESOASSERT(cell_end >= 0 && cell_end <= cell_count, "Bad cell_end");
    this->cell_start = cell_start;
    this->cell_end = cell_end;

    // Make sure the buffer is large enough to store the map.
    const INT npart_total = h_cell_counts_es[cell_end - 1] +
                            static_cast<INT>(h_cell_counts[cell_end - 1]);
    this->d_layer_map->realloc_no_copy(static_cast<std::size_t>(npart_total));
    INT *ptr = this->d_layer_map->ptr;
    INT **h_cell_ptr = this->h_cell_starts->ptr;
    INT **d_cell_ptr = this->d_cell_starts->ptr;
    const std::size_t cell_count_range =
        static_cast<std::size_t>(cell_end - cell_start);
    for (int cx = cell_start; cx < cell_end; cx++) {
      h_cell_ptr[cx] = ptr + h_cell_counts_es[cx];
    }
    this->sycl_target->queue
        .memcpy(d_cell_ptr + cell_start, h_cell_ptr + cell_start,
                cell_count_range * sizeof(INT *))
        .wait_and_throw();
  }

  /**
   * Reset the map.
   */
  inline void reset() {
#ifndef NDEBUG
    this->sycl_target->queue
        .fill<INT *>(this->d_cell_starts->ptr, nullptr, this->cell_count)
        .wait_and_throw();
    this->sycl_target->queue
        .fill<INT *>(this->h_cell_starts->ptr, nullptr, this->cell_count)
        .wait_and_throw();
#endif
    this->cell_start = this->cell_count + 1;
    this->cell_end = -1;
  }
};

} // namespace NESO::Particles

#endif
