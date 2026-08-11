#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_RATE_REDUCTION_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_COLLISION_CELL_RATE_REDUCTION_HPP_

#include "../../compute_target.hpp"
#include "../../device_buffers.hpp"
#include "../../pair_loop/cellwise_pair_list_absolute.hpp"
#include "collision_cell_partition.hpp"

namespace NESO::Particles::DSMC {

/**
 * Implementation to reduce rates on a collision cell basis.
 *
 * Rates are reduced per collision cell as
 *
 *    rate_{cell} = sum_{pair lists} (max_{pairs in cell} (pair reaction rate))
 *
 */
class CollisionCellRateReduction {
protected:
  SYCLTargetSharedPtr sycl_target;
  CollisionCellPartitionSharedPtr collision_cell_partition;
  std::shared_ptr<BufferDevice<int>> d_num_collision_cells;
  std::shared_ptr<BufferDevice<REAL>> d_accumulation_max;

  std::shared_ptr<BufferDevice<REAL>> d_staging_values;
  std::shared_ptr<BufferDevice<int>> d_staging_indices;

  int num_contributors = -1;
  INT num_entries = -1;

  int last_reset_num_contributors = -1;
  int last_reset_num_mesh_cells = -1;
  int last_reset_max_num_collision_cells = -1;

public:
  /// Disable (implicit) copies.
  CollisionCellRateReduction(const CollisionCellRateReduction &st) = delete;
  /// Disable (implicit) copies.
  CollisionCellRateReduction &
  operator=(CollisionCellRateReduction const &a) = delete;

  ~CollisionCellRateReduction() = default;

  /**
   * Create a new reduction object for pairs of particles created using the
   * passed CollisionCellPartition.
   *
   * @param collision_cell_partition CollisionCellPartition from which pairs are
   * sampled.
   */
  CollisionCellRateReduction(
      CollisionCellPartitionSharedPtr collision_cell_partition);

  /**
   * Prepare the internal implementation (reallocate) for N contributing
   * entities, e.g. reactions. Clears all internal state. Contributors are
   * referenced in later calls using an integer in [0, N).
   *
   * @param num_contributors Number of contributors.
   */
  void setup(const int num_contributors);

  /**
   * Resize the internal buffers to be of size num_mesh_cells x
   * max_num_collision_cells as given by the CollisionCellPartition.
   */
  void resize();

  /**
   * Update the internal representation for a contributor by submiting a set of
   * pairs and corresponding rates into the reduction instance. This call will
   * populate the entries in [cell_start, cell_end) of the accumulation buffer
   * for the contributor. For each collision cell the maximum rate supplied in
   * the device rate buffer and the maximum already stored in the internal
   * representation will be stored in the internal representation.
   *
   * @param contributor_id Index of contributor.
   * @param pair_list Pair list describing pairs.
   * @param cell_start First cell of cell block.
   * @param cell_end Last cell plus one of cell block.
   * @param collision_cell_sym ParticleDat containing the collision cell index.
   * @param collision_cell_component ParticleDat component containing the
   * collision cell index.
   * @param device_rate_buffer LocalArray containing a rate per pair in the pair
   * list only for the pairs in [cell_start, cell_end). i.e. This buffer will be
   * indexed using the loop linear index not the pair list linear index.
   */
  void
  update(const int contributor_id,
         CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
         const int cell_start, const int cell_end, Sym<INT> collision_cell_sym,
         const int collision_cell_component,
         LocalArraySharedPtr<REAL> device_rate_buffer);

  /**
   * Update the internal representation for a contributor by submiting a set of
   * pairs and corresponding rates into the reduction instance. This call will
   * populate the entries in [cell_start, cell_end) of the accumulation buffer
   * for the contributor. For each collision cell the maximum rate supplied in
   * the device rate buffer and the maximum already stored in the internal
   * representation will be stored in the internal representation.
   *
   * For mesh cells masked as true in the cell mask vector the supplied initial
   * rate will be used to populate the internal buffer for that contributor
   * rather than the rates indexed by pairs.
   *
   * @param contributor_id Index of contributor.
   * @param pair_list Pair list describing pairs.
   * @param cell_start First cell of cell block.
   * @param cell_end Last cell plus one of cell block.
   * @param collision_cell_sym ParticleDat containing the collision cell index.
   * @param collision_cell_component ParticleDat component containing the
   * collision cell index.
   * @param device_rate_buffer LocalArray containing a rate per pair in the pair
   * list only for the pairs in [cell_start, cell_end). i.e. This buffer will be
   * indexed using the loop linear index not the pair list linear index.
   * @param cell_mask Vector of length num_mesh_cells that is used as a mask for
   * which mesh cells should be set to the initial rate.
   * @param rate_init Rate to set for all collision cells in mesh cells where
   * the mask is non-zero.
   */
  void
  update(const int contributor_id,
         CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
         const int cell_start, const int cell_end, Sym<INT> collision_cell_sym,
         const int collision_cell_component,
         LocalArraySharedPtr<REAL> device_rate_buffer,
         const std::vector<int> &cell_mask, const REAL rate_init);

  /**
   * For mesh cells masked as true in the cell mask vector the supplied initial
   * rate will be used to populate the internal buffer for that contributor
   * rather than the pairs.
   *
   * @param contributor_id Index of contributor.
   * @param cell_mask Vector of length num_mesh_cells that is used as a mask for
   * which mesh cells should be set to the initial rate.
   * @param rate_init Rate to set for all collision cells in mesh cells where
   * the mask is non-zero.
   */
  void update(const int contributor_id, const std::vector<int> &cell_mask,
              const REAL rate_init);

  /**
   * Get the current accumulated rates in a 2D NDLocalArray of size
   * (num_mesh_cells) x (max_num_collision_cells).
   *
   * @param[in, out] accumulated_rates Buffer for reduced rates. Will be
   * allocated if nullptr.
   */
  void get(NDLocalArraySharedPtr<REAL, 2> &accumulated_rates);
};

} // namespace NESO::Particles::DSMC

#endif
