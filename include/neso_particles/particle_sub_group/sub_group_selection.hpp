#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTION_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTION_HPP_

#include "../typedefs.hpp"

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

/**
 * Device copyable type to map from loop cell and loop layer to the actual layer
 * of the particle.
 */
struct MapLoopLayerToLayer {

  /// This member is public but is not part of any API that should be used
  /// outside of NP - use map_loop_layer_to_layer instead.
  // INT const *RESTRICT const *RESTRICT map_ptr;
  INT **map_ptr{nullptr};

  /**
   * For a loop cell and loop layer return the layer of the particle.
   *
   * @param loop_cell Cell containing particle in the selection.
   * @param loop_layer Layer of the particle in the selection.
   * @returns Layer of the particle in the cell.
   */
  template <typename T>
  inline INT map_loop_layer_to_layer(const T loop_cell,
                                     const T loop_layer) const {
    return this->map_ptr[loop_cell][loop_layer];
  }
};

/**
 * Host type that describes a selection of particles.
 */
struct Selection {
  int npart_local{0};
  int ncell{0};
  int *h_npart_cell{nullptr};
  int *d_npart_cell{nullptr};
  INT *d_npart_cell_es{nullptr};
  MapLoopLayerToLayer d_map_cells_to_particles;
};

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
