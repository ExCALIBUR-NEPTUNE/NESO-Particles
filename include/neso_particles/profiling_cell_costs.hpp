#ifndef _NESO_PARTICLES_PROFILING_CELL_COSTS_HPP_
#define _NESO_PARTICLES_PROFILING_CELL_COSTS_HPP_

#include "compute_target.hpp"
#include "device_buffers.hpp"
#include "typedefs.hpp"
#include <vector>

namespace NESO::Particles {

/**
 * The intention of this type is to accumulate and store values that somehow
 * represent the cost of each cell in the mesh.
 */
class ProfilingCellCosts {
protected:
  bool enabled = false;
  std::shared_ptr<BufferDevice<REAL>> d_cell_costs;

public:
  // Compute device
  SYCLTargetSharedPtr sycl_target;

  // Number of cells costs are held for.
  int num_cells;

  /**
   * Create a store for a given number of cells.
   *
   * @param num_cells Number of locally owned mesh cells.
   */
  ProfilingCellCosts(SYCLTargetSharedPtr sycl_target, const int num_cells);

  /**
   * Zero the currently held costs.
   */
  void zero();

  /**
   * Accumulate using a number of particles per cell and a per particle cost.
   *
   * @param d_npart_cell Number of particles in each cell.
   * @param per_particle_cost Estimated cost per particle.
   */
  void accumulate(const REAL *RESTRICT d_npart_cell,
                  const REAL per_particle_cost);
};
} // namespace NESO::Particles

#endif
