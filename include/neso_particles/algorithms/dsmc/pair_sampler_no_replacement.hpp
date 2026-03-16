#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_PAIR_SAMPLER_NO_REPLACEMENT_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_PAIR_SAMPLER_NO_REPLACEMENT_HPP_

#include "../../compute_target.hpp"
#include "../../containers/rng/host_rng_common.hpp"
#include "../../pair_loop/cellwise_pair_list.hpp"
#include "collision_cell_partition.hpp"
#include <memory>
#include <vector>

namespace NESO::Particles::DSMC {

/**
 * Class for sampling pairs of particles without replacement within a collision
 * cell.
 */
class PairSamplerNoReplacement : public CellwisePairList {
protected:
  std::vector<int> h_wave_count;
  std::unique_ptr<BufferDevice<int>> d_wave_count;

  std::unique_ptr<BufferDevice<int>> d_wave_offsets;
  std::unique_ptr<CellDat<int>> d_pair_list;

  std::vector<int> h_pair_counts;
  std::unique_ptr<BufferDevice<int>> d_pair_counts;
  std::vector<INT> h_pair_counts_es;
  std::unique_ptr<BufferDevice<INT>> d_pair_counts_es;

  std::vector<int> h_num_collision_cells;
  std::unique_ptr<BufferDevice<int>> d_num_collision_cells;
  INT num_pairs{0};

  CellwisePairListDevice d_pair_list_device;

  std::unique_ptr<BufferDevice<INT>> d_max_pair_count;

public:
  /// Disable (implicit) copies.
  PairSamplerNoReplacement(const PairSamplerNoReplacement &st) = delete;
  /// Disable (implicit) copies.
  PairSamplerNoReplacement &
  operator=(PairSamplerNoReplacement const &a) = delete;

  virtual ~PairSamplerNoReplacement() = default;

  // Compute device
  SYCLTargetSharedPtr sycl_target;

  // Number of mesh cells
  int cell_count{0};

  // RNG Generation function.
  std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function;

  /**
   * Create a sampler for a given compute target and mesh cell count.
   *
   * @param sycl_target Compute device to use.
   * @param cell_count Mesh cell count.
   * @param rng_generation_function Source of random samples.
   */
  PairSamplerNoReplacement(
      SYCLTargetSharedPtr sycl_target, const int cell_count,
      std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function);

  /**
   * Sample pairs in each collision cell between species A and B.
   *
   * @param collision_cell_partition Map from collision cells to particles
   * within those collision cells.
   * @param species_id_a First species ID.
   * @param species_id_b Second species ID.
   * @param map_cells_to_counts Map from mesh cell to counts for each collision
   * cell.
   */
  void sample(CollisionCellPartitionSharedPtr collision_cell_partition,
              const INT species_id_a, const INT species_id_b,
              const std::vector<std::vector<int>> &map_cells_to_counts);

  /**
   * Get a description of the pair list accessible on the device.
   *
   * @returns PairListDevice describing all the particle pairs. The returned
   * object must have a lifetime equal or shorter than this host instance.
   */
  virtual CellwisePairListDevice get_pair_list() override;

  /**
   * Get a description of the pair list accessible on the host.
   *
   * @returns Container of pairs for each cell.
   */
  virtual CellwisePairListHostMap get_host_pair_list() override;
};

} // namespace NESO::Particles::DSMC

#endif
