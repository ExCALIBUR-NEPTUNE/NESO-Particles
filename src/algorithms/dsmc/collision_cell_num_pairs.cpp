#include <neso_particles/algorithms/dsmc/collision_cell_num_pairs.hpp>

namespace NESO::Particles::DSMC {

CollisionCellNumPairs::CollisionCellNumPairs(SYCLTargetSharedPtr sycl_target,
                                             const int num_mesh_cells,
                                             const int max_num_collision_cells)
    : h_entries(std::vector<int, HostAllocator<int>>(
          num_mesh_cells * max_num_collision_cells,
          HostAllocator<int>(sycl_target->queue))),
      num_mesh_cells(num_mesh_cells),
      max_num_collision_cells(max_num_collision_cells)

{}

int &CollisionCellNumPairs::at(const int mesh_cell, const int collision_cell) {

  NESOASSERT((0 <= mesh_cell) && (mesh_cell < this->num_mesh_cells),
             "Bad mesh cell passed.");

  return this
      ->h_entries[mesh_cell * this->max_num_collision_cells + collision_cell];
}

} // namespace NESO::Particles::DSMC
