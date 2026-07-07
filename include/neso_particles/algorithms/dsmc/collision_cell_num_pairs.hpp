#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_NUM_PAIRS_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_NUM_PAIRS_HPP_

#include "../../compute_target.hpp"
#include <memory>
#include <vector>

namespace NESO::Particles::DSMC {

class CollisionCellPartition;

/**
 * Container for and integer value for all collision cells in the local domain.
 * E.g. maximum number of pairs and number of pairs to sample.
 */
class CollisionCellNumPairs {

protected:
  std::vector<int, HostAllocator<int>> h_entries;

public:
  // The number of mesh cells this container was created with.
  int num_mesh_cells;
  // The max number of collision cells this container considers.
  int max_num_collision_cells;

  /**
   * Create an instance for a number of mesh cells and maximum number of
   * collision cells across all mesh cells.
   *
   * @param sycl_target Corresponding compute device.
   * @param num_mesh_cells Number of mesh cells.
   * @param max_num_collision_cells Maximum number of collision cells in any
   * mesh cell.
   */
  CollisionCellNumPairs(SYCLTargetSharedPtr sycl_target,
                        const int num_mesh_cells,
                        const int max_num_collision_cells);

  /**
   * Reference the held values for a mesh cell and collision cell.
   *
   * @param mesh_cell Mesh cell to access value for.
   * @param collision_cell Collision cell to access value for.
   * @returns Modifiable reference to value.
   */
  int &at(const int mesh_cell, const int collision_cell);

  /**
   * @returns Host pointer to map. Data is stored mesh cells slowest then
   * collision cells.
   */
  int *get_host_pointer();

  /**
   * Set all entries to the same value.
   *
   * @param value Value to set.
   */
  void fill(const int value);
};

using CollisionCellNumPairsSharedPtr = std::shared_ptr<CollisionCellNumPairs>;

} // namespace NESO::Particles::DSMC

#endif
