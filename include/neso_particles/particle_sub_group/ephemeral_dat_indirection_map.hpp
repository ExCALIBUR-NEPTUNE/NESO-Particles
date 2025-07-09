#ifndef _NESO_PARTICLES__PARTICLE_SUB_GROUP_EPHEMERAL_DAT_INDIRECTION_MAP_HPP_
#define _NESO_PARTICLES__PARTICLE_SUB_GROUP_EPHEMERAL_DAT_INDIRECTION_MAP_HPP_

#include "../compute_target.hpp"
#include "../device_buffers.hpp"
#include "sub_group_selection.hpp"

namespace NESO::Particles {

class EphemeralDatIndirectionMap {
protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  struct IndirectionMap {
    std::shared_ptr<BufferDevice<INT>> d_map{nullptr};
    std::shared_ptr<BufferDevice<INT *>> d_cell_entry_points{nullptr};
  };

  IndirectionMap identity_map;

  inline void init_cell_entry_points(IndirectionMap *indirection_map) {
    if (indirection_map->d_cell_entry_points == nullptr) {
      indirection_map->d_cell_entry_points =
          std::make_shared<BufferDevice<INT *>>(this->sycl_target,
                                                this->cell_count);
    }
  }

  inline void init_map(IndirectionMap *indirection_map, const int npart) {
    if (indirection_map->d_map == nullptr) {
      indirection_map->d_map =
          std::make_shared<BufferDevice<INT>>(this->sycl_target, npart);
    } else {
      indirection_map->d_map->realloc_no_copy(npart);
    }
  }

  inline void create_identity_map(const int npart) {
    INT *k_map = this->identity_map.d_map->ptr;
    auto e0 = this->sycl_target->queue.parallel_for(
        sycl::range<1>(npart), [=](auto ix) { k_map[ix] = ix; });

    INT **k_cell_entry_points = this->identity_map.d_cell_entry_points->ptr;
    auto e1 = this->sycl_target->queue.fill(
        k_cell_entry_points, k_map, static_cast<std::size_t>(this->cell_count));

    e0.wait_and_throw();
    e1.wait_and_throw();
  }

public:
  // Compute device to create indirection maps on.
  SYCLTargetSharedPtr sycl_target;
  // Cell count of domain.
  int cell_count;

  /**
   * Construct a new set of indirection maps.
   *
   * @param sycl_target Compute device for maps.
   * @param cell_count Number of cells in the domain.
   */
  EphemeralDatIndirectionMap(SYCLTargetSharedPtr sycl_target,
                             const int cell_count)
      : sycl_target(sycl_target), cell_count(cell_count) {
    this->reset(8);
  }

  /**
   * Reset the container, e.g. after hybrid/cell_move.
   *
   * @param max_cell_occupancy Maximum occupancy of any cell in the domain.
   */
  inline void reset(const int max_cell_occupancy) {
    this->init_cell_entry_points(&this->identity_map);
    if ((this->identity_map.d_map == nullptr) ||
        (this->identity_map.d_map->size < max_cell_occupancy)) {
      this->init_map(&this->identity_map, max_cell_occupancy);
      this->create_identity_map(max_cell_occupancy);
    }
  }
};

} // namespace NESO::Particles

#endif
