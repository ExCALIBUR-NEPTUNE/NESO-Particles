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
  std::shared_ptr<BufferDevice<REAL>> d_accumulation_max;
  std::shared_ptr<BufferDevice<REAL>> d_accumulation_plus;

  std::shared_ptr<BufferDevice<REAL>> d_staging_values;
  std::shared_ptr<BufferDevice<int>> d_staging_indices;

  sycl::event event_fill_plus;
  sycl::event event_reduce_max;
  sycl::event event_reduce_plus;

  INT num_entries = -1;
  int cell_start = -1;
  int cell_end = -1;

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
   * Defines the cell range([cell_start, cell_end)) over which reductions will
   * be performed and resets the accumulation buffer to zero.
   *
   * @param cell_start First cell of cell block.
   * @param cell_end Last cell plus one of cell block.
   */
  void reset(const int cell_start, const int cell_end);

  /**
   * Submit a set of pairs and corresponding rates into the reduction instance.
   *
   * @param pair_list Pair list describing pairs.
   * @param collision_cell_sym ParticleDat containing the collision cell index.
   * @param collision_cell_component ParticleDat component containing the
   * collision cell index.
   * @param device_rate_buffer LocalArray containing a rate per pair in the pair
   * list.
   */
  void
  submit(CellwisePairListAbsolute<ParticleGroup, CellwisePairList> &pair_list,
         Sym<INT> collision_cell_sym, const int collision_cell_component,
         LocalArraySharedPtr<REAL> device_rate_buffer);

  /**
   * Get the current accumulated rates in a 2D NDLocalArray of size (cell_end -
   * cell_start) x (max_num_collision_cells).
   *
   * @param[in, out] accumulated_rates Buffer for reduced rates. Will be
   * allocated if nullptr.
   */
  void get(NDLocalArraySharedPtr<REAL, 2> &accumulated_rates);
};

} // namespace NESO::Particles::DSMC

#endif
