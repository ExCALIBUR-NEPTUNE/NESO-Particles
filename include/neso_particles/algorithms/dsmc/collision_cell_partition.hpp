#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_PARTITION_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_PARTITION_HPP_

#include "../../compute_target.hpp"
#include "../../containers/blocked_binary_tree.hpp"
#include "../../particle_sub_group/particle_sub_group.hpp"
#include <memory>
#include <vector>

namespace NESO::Particles::DSMC {

/**
 * TODO
 */
class CollisionCellPartition {
protected:
  // This is the map from the species IDs the user specified to the linear index
  // we use in the data structures.
  std::unique_ptr<BlockedBinaryTree<INT, INT>> h_map_species_id_linear_id;
  INT num_species{0};

public:
  /// Disable (implicit) copies.
  CollisionCellPartition(const CollisionCellPartition &st) = delete;
  /// Disable (implicit) copies.
  CollisionCellPartition &operator=(CollisionCellPartition const &a) = delete;
  ~CollisionCellPartition() = default;

  // Compute device
  SYCLTargetSharedPtr sycl_target;

  // Number of mesh cells
  int cell_count{0};

  // Permissible species IDs.
  std::vector<INT> species_ids;

  // Current number of collision cells for each mesh cell.
  std::vector<int> collision_cell_counts;

  /**
   * Create a container that holds a representation of particles partitioned
   * into mesh cells then DSMC collision cells.
   *
   * @param sycl_target SYCLTarget to use.
   * @param cell_count Number of local mesh cells (not the number of collision
   * cells).
   * @param species_ids Vector containing all permissible species IDs that could
   * be encountered.
   */
  CollisionCellPartition(SYCLTargetSharedPtr sycl_target, const int cell_count,
                         std::vector<INT> species_ids);

  /**
   * TODO
   *
   * @param particle_sub_group The set of particles which are partitioned into
   * species and dsmc cells.
   * @param collision_cell_counts Vector of length cell_count containing the
   * number of collision cells in each mesh cell.
   */
  void construct(ParticleSubGroupSharedPtr particle_sub_group,
                 std::vector<int> &collision_cell_counts,
                 Sym<INT> species_id_sym, const int species_id_component,
                 Sym<INT> collision_cell_sym,
                 const int collision_cell_component);
};

} // namespace NESO::Particles::DSMC

#endif
