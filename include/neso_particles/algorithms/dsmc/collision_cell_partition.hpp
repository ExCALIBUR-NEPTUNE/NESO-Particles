#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_PARTITION_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_PARTITION_HPP_

#include "../../compute_target.hpp"
#include "../../containers/blocked_binary_tree.hpp"
#include "../../containers/nd_local_array.hpp"
#include "../../containers/particle_mask.hpp"
#include "../../nd_host_array.hpp"
#include "../../particle_sub_group/particle_sub_group.hpp"
#include "collision_cell_num_pairs.hpp"
#include <memory>
#include <vector>

namespace NESO::Particles::DSMC {

/**
 * Device type corresponding to CollisionCellPartition.
 */
struct CollisionCellPartitionDevice {
  INT const *d_collision_cell_offsets{nullptr};

  BlockedBinaryNode<INT, INT, NESO_PARTICLES_BLOCKED_BINARY_TREE_WIDTH>
      *d_species_map_root{nullptr};

  INT num_mesh_cells{0};
  INT max_num_collision_cells{0};
  INT max_num_species{0};
  int const *d_map_entries;

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
  inline INT get_num_particles_cell_species(const INT cell_mesh,
                                            const INT cell_collision,
                                            const INT linear_species_id) const {
    const INT offset = this->get_offset_cell_species(cell_mesh, cell_collision,
                                                     linear_species_id);
    const INT npart_rhs = this->d_collision_cell_offsets[offset + 1];
    const INT npart_lhs = this->d_collision_cell_offsets[offset];
    return npart_rhs - npart_lhs;
  }

  /**
   * @param cell_mesh Mesh cell to index collision cell.
   * @param cell_collision Collision cell index within the mesh cell.
   * @param linear_species_id Linear species index.
   * @returns The number of particles in the requested collision cell with the
   * requested linear species.
   */
  inline INT get_offset_cell_species(const INT cell_mesh,
                                     const INT cell_collision,
                                     const INT linear_species_id) const {

    const INT stride_mesh_cells = max_num_collision_cells * max_num_species;
    const INT offset_cell = cell_mesh * stride_mesh_cells;

    const INT offset =
        offset_cell + cell_collision * max_num_species + linear_species_id;

    return offset;
  }

  /**
   * @param cell_mesh Mesh cell to index collision cell.
   * @param cell_collision Collision cell index within the mesh cell.
   * @param linear_species_id Linear species index.
   * @param particle_index Linear index of particle within the species and
   * collision cell.
   * @returns Layer of particle within the mesh cell.
   */
  inline INT get_particle_layer(const INT cell_mesh, const INT cell_collision,
                                const INT linear_species_id,
                                const INT particle_index) const {
    const INT index = this->get_offset_cell_species(cell_mesh, cell_collision,
                                                    linear_species_id);
    const INT offset = this->d_collision_cell_offsets[index];
    return this->d_map_entries[offset + particle_index];
  }
};

/**
 * Holds a map from (mesh cell, collision cell, species ID) to particle layers.
 * Particles may be masked off by passing a ParticleMask to the construct call.
 * Particles exist in the map independently of the mask values.
 */
class CollisionCellPartition {
protected:
  // This is the map from the species IDs the user specified to the linear index
  // we use in the data structures.
  std::unique_ptr<BlockedBinaryTree<INT, INT>> h_map_species_id_linear_id;
  INT num_species{0};
  std::unique_ptr<BufferDevice<INT>> d_collision_cell_offsets;
  std::unique_ptr<BufferDevice<int>> d_map_entries;

  std::map<INT, INT> map_species_id_to_linear;

  std::unique_ptr<BufferDevice<INT>> d_max_collision_cell_occupancy;
  ParticleMaskSharedPtr particle_mask = nullptr;

public:
  /// Disable (implicit) copies.
  CollisionCellPartition(const CollisionCellPartition &st) = delete;
  /// Disable (implicit) copies.
  CollisionCellPartition &operator=(CollisionCellPartition const &a) = delete;
  ~CollisionCellPartition() = default;

  // Compute device
  SYCLTargetSharedPtr sycl_target;

  // Number of mesh cells
  int num_mesh_cells{0};

  // The ParticleSubGroup from the last construct.
  ParticleSubGroupSharedPtr particle_sub_group;

  // Permissible species IDs.
  std::vector<INT> species_ids;

  // Current number of collision cells for each mesh cell.
  std::vector<int> num_collision_cells;

  // Maximum number of collision cells over all mesh cells.
  INT max_num_collision_cells{0};

  // Maximum occupancy of a collision cell for any species. This value ignores
  // values of masks.
  INT max_collision_cell_occupancy{0};

  /**
   * Create a container that holds a representation of particles partitioned
   * into mesh cells then DSMC collision cells.
   *
   * @param sycl_target SYCLTarget to use.
   * @param num_mesh_cells Number of local mesh cells (not the number of
   * collision cells).
   * @param species_ids Vector containing all permissible species IDs that could
   * be encountered.
   */
  CollisionCellPartition(SYCLTargetSharedPtr sycl_target,
                         const int num_mesh_cells,
                         std::vector<INT> species_ids);

  /**
   * Construct the internal representation from a ParticleSubGroup. This
   * construct method enables all particles, i.e. there are no masks values
   * used.
   *
   * @param particle_sub_group The set of particles which are partitioned into
   * species and dsmc cells.
   * @param num_collision_cells Vector of length num_mesh_cells containing the
   * number of collision cells in each mesh cell.
   * @param species_id_sym Sym describing which ParticleDat contains the species
   * ID.
   * @param species_id_component Describe which ParticleDat component describes
   * the species.
   * @param collision_cell_sym Sym describing which ParticleDat contains the
   * collision cell ID.
   * @param collision_cell_component Describe which ParticleDat component
   * contains the collision cell ID.
   */
  void construct(ParticleSubGroupSharedPtr particle_sub_group,
                 const std::vector<int> &num_collision_cells,
                 Sym<INT> species_id_sym, const int species_id_component,
                 Sym<INT> collision_cell_sym,
                 const int collision_cell_component);

  /**
   * Construct the internal representation from a ParticleSubGroup. Particles
   * may be masked off, or on, from pair selection after construct is called by
   * updating the ParticleMask that is passed here. Particles exist in this data
   * structure indpendently of the mask value. The current value of the mask is
   * taken into account when get_max_num_pairs is called.
   *
   * @param particle_sub_group The set of particles which are partitioned into
   * species and dsmc cells.
   * @param particle_mask Particle masks that indicate if a particle is
   * available.
   * @param num_collision_cells Vector of length num_mesh_cells containing the
   * number of collision cells in each mesh cell.
   * @param species_id_sym Sym describing which ParticleDat contains the species
   * ID.
   * @param species_id_component Describe which ParticleDat component describes
   * the species.
   * @param collision_cell_sym Sym describing which ParticleDat contains the
   * collision cell ID.
   * @param collision_cell_component Describe which ParticleDat component
   * contains the collision cell ID.
   */
  void construct(ParticleSubGroupSharedPtr particle_sub_group,
                 ParticleMaskSharedPtr particle_mask,
                 const std::vector<int> &num_collision_cells,
                 Sym<INT> species_id_sym, const int species_id_component,
                 Sym<INT> collision_cell_sym,
                 const int collision_cell_component);

  /**
   * Determine the maximum number of pairs that can be formed between species A
   * and B for each collision cell. Note that in the case of replacement these
   * values are identical to determining if there are two or more particles in
   * the collision cell. In this scenario the return values are 0 or INT_MAX.
   * This method will use the current values of the ParticleMask provided if a
   * ParticleMask was provided at construction time.
   *
   * @param[in] species_id_a Species ID of A.
   * @param[in] species_id_b Species ID of B.
   * @param[in] replacement Indicate if pairs are chosen with (true) or without
   * replacement (false).
   * @param[in, out] map_cell_to_num_pairs Output map from [mesh cell][collision
   * cell] to maximum number of pairs that can be formed.
   */
  void get_max_num_pairs(const INT species_id_a, const INT species_id_b,
                         const bool replacement,
                         CollisionCellNumPairsSharedPtr &map_cell_to_num_pairs);

  /**
   * @param species_id Species ID as stored on particles.
   * @returns Linear species ID as used in the device maps.
   */
  INT get_linear_species_id(const INT species_id);

  /**
   * @returns The device description of the maps.
   */
  CollisionCellPartitionDevice get_device();

  /**
   * @returns A CollisionCellNumPairs instance suitably sized for the number of
   * mesh cells and collision cells currently specified.
   */

  CollisionCellNumPairsSharedPtr get_collision_cell_num_pairs_instance();

  /**
   * @returns The current ParticleMask in use. This method may return a nullptr.
   */
  ParticleMaskSharedPtr get_particle_mask();

  /**
   * Get the number of unmasked particles of a particular species ID in each
   * collision cell.
   *
   * @param[in] species_id Species ID of particles to get number of.
   * @param[in, out] num_particles NDHostArray of particle numbers per collision
   * cell. Should be of dimension 2 with extents num_mesh_cells and
   * max_num_collision_cells. Will be allocated if nullptr.
   */
  void get_num_unmasked_particles(const INT species_id,
                                  NDHostArraySharedPtr<int, 2> &num_particles);

  /**
   * Get the number of unmasked particles of a particular species ID in each
   * collision cell.
   *
   * @param[in] species_id Species ID of particles to get number of.
   * @param[in, out] num_particles NDLocalArray of particle numbers per
   * collision cell. Should be of dimension 2 with extents num_mesh_cells and
   * max_num_collision_cells. Will be allocated if nullptr.
   */
  void get_num_unmasked_particles(const INT species_id,
                                  NDLocalArraySharedPtr<int, 2> &num_particles);
};

using CollisionCellPartitionSharedPtr = std::shared_ptr<CollisionCellPartition>;

} // namespace NESO::Particles::DSMC

#endif
