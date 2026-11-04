#ifndef _NESO_PARTICLES_PROFILING_CELL_COSTS_HPP_
#define _NESO_PARTICLES_PROFILING_CELL_COSTS_HPP_

#include "compute_target.hpp"
#include "device_buffers.hpp"
#include "typedefs.hpp"
#include <cstring>
#include <vector>

namespace NESO::Particles {

/**
 * The intention of this type is to accumulate and store values that somehow
 * represent the cost of each cell in the mesh.
 */
class ProfilingCellCosts {
protected:
  bool enabled = false;
  std::vector<REAL> h_cell_costs;
  std::vector<int> h_npart_cell;

public:
  // Number of cells costs are held for.
  int num_cells;

  /**
   * Create a store for a given number of cells.
   *
   * @param num_cells Number of locally owned mesh cells.
   */
  ProfilingCellCosts(const int num_cells);

  /**
   * Enable recording of costs.
   */
  void enable();

  /**
   * Disable recording of costs.
   */
  void disable();

  /**
   * Zero the currently held costs.
   */
  void zero();

  /**
   * Accumulate using a number of particles per cell and a per particle cost.
   *
   * @param sycl_target Compute device.
   * @param d_npart_cell Number of particles in each cell.
   * @param per_particle_cost Estimated cost per particle.
   */
  void accumulate(SYCLTargetSharedPtr sycl_target,
                  const int *RESTRICT d_npart_cell,
                  const REAL per_particle_cost);

  /**
   * Get the per cell estimated costs.
   *
   * @param[in, out] cell_costs Output vector to be populated with cell costs.
   */
  void get_cell_costs(std::vector<REAL> &cell_costs);
};
} // namespace NESO::Particles

#endif
