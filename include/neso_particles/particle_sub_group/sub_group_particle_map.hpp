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
  // The number of particles in the map.
  INT npart_total;

  // Helper buffers that the user will need to create the Selection and map.
  // These buffers will not be modified internally.
  std::shared_ptr<BufferDeviceHost<int>> dh_npart_cell;
  std::shared_ptr<BufferDeviceHost<INT>> dh_npart_cell_es;

  /**
   * @returns the host and device pointers for dh_npart_cell and
   * dh_npart_cell_es.
   */
  std::tuple<int *, int *, INT *, INT *> get_helper_ptrs();

  /**
   * @param cell_count The number of cells to create a map for.
   */
  SubGroupParticleMap(SYCLTargetSharedPtr sycl_target, const int cell_count);

  /**
   * Create a map for a range of cells.
   */
  void create(const int cell_start, const int cell_end,
              const int *RESTRICT const h_cell_counts,
              const INT *RESTRICT const h_cell_counts_es);

  /**
   * Reset the map.
   */
  void reset();
};

} // namespace NESO::Particles

#endif
