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
struct CollisionCellPartitionDevice {
  INT *d_collision_cell_offsets{nullptr};

  BlockedBinaryNode<INT, INT, NESO_PARTICLES_BLOCKED_BINARY_TREE_WIDTH>
      *d_species_map_root{nullptr};

  INT mesh_cell_count{0};
  INT max_num_collision_cells{0};
  INT max_num_species{0};

  /**
   * @param species_id Input species ID from user space.
   * @param linear_species_id Output linearised species id.
   * @returns True if linear species ID found otherwise false.
   */
  inline bool get_linear_species_index(const INT species_id,
                                       INT *linear_species_id) const {
    return this->d_species_map_root->get(species_id, linear_species_id);
  }

  /**
   * @param cell_mesh Mesh cell to index collision cell.
   * @param cell_collision Collision cell index within the mesh cell.
   * @param linear_species_id Linear species index.
   * @returns The number of particles in the requested collision cell with the
   * requested linear species.
   */
  inline INT get_num_particles_cell_species(const int cell_mesh,
                                            const int cell_collision,
                                            const INT linear_species_id) const {
    const INT offset = this->get_offset_cell_species(cell_mesh, cell_collision,
                                                     linear_species_id);
    const INT npart_rhs = d_collision_cell_offsets[offset + 1];
    const INT npart_lhs = d_collision_cell_offsets[offset];
    return npart_rhs - npart_lhs;
  }

  /**
   * @param cell_mesh Mesh cell to index collision cell.
   * @param cell_collision Collision cell index within the mesh cell.
   * @param linear_species_id Linear species index.
   * @returns The number of particles in the requested collision cell with the
   * requested linear species.
   */
  inline INT get_offset_cell_species(const int cell_mesh,
                                     const int cell_collision,
                                     const INT linear_species_id) const {

    const INT stride_mesh_cells = max_num_collision_cells * max_num_species;
    const INT offset_cell = cell_mesh * stride_mesh_cells;

    const INT offset =
        offset_cell + cell_collision * max_num_species + linear_species_id;

    return offset;
  }
};

/**
 * TODO
 */
class CollisionCellPartition {
protected:
  // This is the map from the species IDs the user specified to the linear index
  // we use in the data structures.
  std::unique_ptr<BlockedBinaryTree<INT, INT>> h_map_species_id_linear_id;
  INT num_species{0};
  std::unique_ptr<BufferDevice<INT>> d_collision_cell_offsets;

  INT max_num_collision_cells{0};

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

  /**
   * @returns The device description of the maps.
   */
  CollisionCellPartitionDevice get_device();
};

} // namespace NESO::Particles::DSMC

#endif
