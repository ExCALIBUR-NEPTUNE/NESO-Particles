#ifndef _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_
#define _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_

#include "../compute_target.hpp"
#include "../sycl_typedefs.hpp"
#include "../typedefs.hpp"

namespace NESO::Particles {
template <typename T> class ParticleDatT;
}

namespace NESO::Particles::ParticleLoopImplementation {

/**
 * For a set of cells containing particles create several sycl::nd_range
 * instances which cover the iteration space of all particles. This exists to
 * create an iteration set over all particles which is blocked, to reduce the
 * number of kernel launches, and reasonably robust to non-uniform.
 */
struct ParticleLoopIterationSet {

  /// The number of blocks of cells.
  const int nbin;
  /// The number of cells.
  const int ncell;
  /// Host accessible pointer to the number of particles in each cell.
  int *h_npart_cell;
  /// Container to store the sycl::nd_ranges.
  std::vector<sycl::nd_range<2>> iteration_set;
  /// Offsets to add to the cell index to map to the correct cell.
  std::vector<std::size_t> cell_offsets;
  /// The size of the last iteration set computed
  std::size_t iteration_set_size;

  /**
   *  Creates iteration set creator for a given set of cell particle counts.
   *
   *  @param nbin Number of blocks of cells.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   */
  ParticleLoopIterationSet(const int nbin, const int ncell, int *h_npart_cell);

  /**
   *  Create and return an iteration set which is formed as nbin
   *  sycl::nd_ranges.
   *
   *  @param cell If set iteration set will only cover this cell.
   *  @param local_size Optional size of SYCL work groups.
   *  @returns Tuple containing: Number of bins, sycl::nd_ranges, cell index
   *  offsets.
   */
  std::tuple<int, std::vector<sycl::nd_range<2>> &, std::vector<std::size_t> &>
  get(const std::optional<int> cell = std::nullopt,
      const size_t local_size = 256);
};

/**
 * Type to simplify determining which cell and layer an nd_item is working on.
 */
struct ParticleLoopBlockDevice {
  std::size_t offset_cell;
  std::size_t offset_layer;
  int const *RESTRICT d_npart_cell;
  std::size_t stride;

  /**
   * Convert a sycl::nd_item<2> into a cell, layer pair when in strided mode but
   * work items access particles in an interlaced mode not a block mode.
   * @param[in] idx SYCL nd_item for work item.
   * @param[in] stride_index Which index in the stride is this.
   * @param[in, out] cell Cell for iteration.
   * @param[in, out] layer Layer for iteration.
   */
  inline void get_interlaced_cell_layer(const sycl::nd_item<2> &idx,
                                        const std::size_t stride_index,
                                        std::size_t *RESTRICT cell,
                                        std::size_t *RESTRICT layer) const {
    *cell = idx.get_global_id(0) + this->offset_cell;
    *layer = this->offset_layer * this->stride +
             idx.get_group().get_group_id(1) * this->stride *
                 idx.get_local_range(1) +
             stride_index * idx.get_local_range(1) + idx.get_local_id(1);
  }

  /**
   * Convert a sycl::nd_item<2> into a cell, layer pair.
   * @param[in] idx SYCL nd_item for work item.
   * @param[in, out] cell Cell for iteration.
   * @param[in, out] layer Layer for iteration.
   */
  inline void get_cell_layer(const sycl::nd_item<2> &idx,
                             std::size_t *RESTRICT cell,
                             std::size_t *RESTRICT layer) const {
    *cell = idx.get_global_id(0) + this->offset_cell;
    *layer = idx.get_global_id(1) + this->offset_layer;
  }

  /**
   * @param cell Cell for this work item, see get_cell_layer.
   * @param layer Layer for this work item, see get_cell_layer.
   * @returns True if work item should operate on particles. Valid when stride
   * == 1.
   */
  inline bool work_item_required(const std::size_t cell,
                                 const std::size_t layer) const {
    return layer < static_cast<std::size_t>(this->d_npart_cell[cell]);
  }

  /**
   * Convert a sycl::nd_item<2> into a cell, block pair for the first particle.
   * @param[in] idx SYCL nd_item for work item.
   * @param[in, out] cell Cell for iteration.
   * @param[in, out] block Block for iteration.
   */
  inline void stride_get_cell_block(const sycl::nd_item<2> &idx,
                                    std::size_t *RESTRICT cell,
                                    std::size_t *RESTRICT block) const {
    *cell = idx.get_global_id(0) + this->offset_cell;
    *block = idx.get_global_id(1) + this->offset_layer;
  }

  /**
   * @param cell Cell for this work item, see stride_get_cell_block.
   * @param block Block for this work item, see stride_get_cell_block.
   * @returns True if work item should operate on particles. Valid when stride
   * != 1.
   */
  inline bool stride_work_item_required(const std::size_t cell,
                                        const std::size_t block) const {
    return (block * stride) <
           static_cast<std::size_t>(this->d_npart_cell[cell]);
  }

  /**
   * @param cell Cell for this work item, see stride_get_cell_block.
   * @param block Block for this work item, see stride_get_cell_block.
   * @returns Local index of last workitem which can touch particle data plus
   * one.
   */
  inline std::size_t stride_local_index_bound(const std::size_t cell,
                                              const std::size_t block) const {
    const std::size_t last_index =
        sycl::min((block + 1) * stride,
                  static_cast<std::size_t>(this->d_npart_cell[cell])) -
        block * stride;
    return last_index;
  }

  ParticleLoopBlockDevice() = default;
};
static_assert(std::is_trivially_copyable<ParticleLoopBlockDevice>::value ==
              true);

/**
 * Type to wrap the parallel_for iteration set specification and the device
 * type which computes the particle cell and layer.
 */
struct ParticleLoopBlockHost {
  /// Device copyable type that describes the loop bounds for the kernel.
  ParticleLoopBlockDevice block_device;
  /// Does the kernel need to check the loop bounds for the layers in each
  /// cell.
  bool layer_bounds_check_required{true};
  /// The actual size of the local range used.
  std::size_t local_size;
  /// The iteration set for the parallel loop
  sycl::nd_range<2> loop_iteration_set;

  ParticleLoopBlockHost(ParticleLoopBlockDevice block_device,
                        const bool layer_bounds_check_required,
                        const std::size_t local_size,
                        sycl::nd_range<2> loop_iteration_set)
      : block_device(block_device),
        layer_bounds_check_required(layer_bounds_check_required),
        local_size(local_size), loop_iteration_set(loop_iteration_set) {}
};

/**
 * Type to create iteration sets that implement a particle loop.
 */
class ParticleLoopBlockIterationSet {
protected:
  /// SYCL device
  SYCLTargetSharedPtr sycl_target;
  /// The number of cells.
  const std::size_t ncell;
  /// Host accessible pointer to the number of particles in each cell.
  int *h_npart_cell;
  /// Device accessible pointer to the number of particles in each cell.
  int *d_npart_cell;

  std::size_t get_global_size(const std::size_t N,
                              const std::size_t local_size);

  std::size_t
  get_local_size(std::size_t local_size,
                 const std::size_t num_bytes_local, // this is per particle
                 const std::size_t stride);

public:
  /// The last iteration set produced
  std::vector<ParticleLoopBlockHost> iteration_set;
  /// The size of the last iteration set computed
  std::size_t iteration_set_size;

  /**
   *  Creates iteration set creator for a given set of cell particle counts.
   *
   *  @param sycl_target Compute device to use.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   *  @param d_npart_cell Device accessible array of cell particle counts.
   */
  ParticleLoopBlockIterationSet(SYCLTargetSharedPtr sycl_target,
                                const std::size_t ncell, int *h_npart_cell,
                                int *d_npart_cell);

  /**
   *  Creates iteration set creator for a given ParticleDat.
   *
   *  @param particle_dat Specify an iteration set from a ParticleDat.
   */
  ParticleLoopBlockIterationSet(
      std::shared_ptr<ParticleDatT<REAL>> particle_dat);

  /**
   *  Creates iteration set creator for a given ParticleDat.
   *
   *  @param particle_dat Specify an iteration set from a ParticleDat.
   */
  ParticleLoopBlockIterationSet(
      std::shared_ptr<ParticleDatT<INT>> particle_dat);

  /**
   * Get a complete iteration set for a particle loop for all cells in the
   * passed range.
   *
   * @param cell_start First cell.
   * @param cell_end Last cell to visit + 1.
   * @param nbin Default number of bins to use for kernel launch.
   * @param local_size Default local size to use for kernel launch.
   * @param num_bytes_local Number of bytes required per particle.
   * @param stride Number of particles each work item will process, default 1.
   */
  std::vector<ParticleLoopBlockHost> &
  get_generic(const std::size_t cell_start, const std::size_t cell_end,
              std::size_t nbin = 16, std::size_t local_size = 256,
              const std::size_t num_bytes_local = 0,
              const std::size_t stride = 1);

  /**
   * Get a complete iteration set for a particle loop for all cells.
   *
   * @param nbin Default number of bins to use for kernel launch.
   * @param local_size Default local size to use for kernel launch.
   * @param num_bytes_local Number of bytes required per particle.
   * @param stride Number of particles each work item will process, default 1.
   */
  std::vector<ParticleLoopBlockHost> &
  get_all_cells(std::size_t nbin = 16, std::size_t local_size = 256,
                const std::size_t num_bytes_local = 0,
                const std::size_t stride = 1);

  /**
   * Get an iteration set for a particle loop for a single cell.
   *
   * @param cell Produce an iteration set for a single cell.
   * @param local_size Default local size to use for kernel launch.
   * @param num_bytes_local Number of bytes required per particle.
   * @param stride Number of particles each work item will process, default 1.
   */
  std::vector<ParticleLoopBlockHost> &
  get_single_cell(const std::size_t cell, std::size_t local_size = 256,
                  const std::size_t num_bytes_local = 0,
                  const std::size_t stride = 1);

  /**
   * Get an iteration set for a range of cells.
   *
   * @param cell_start Starting cell.
   * @param cell_end Bounding cell (last cell to visit +1).
   * @param local_size Default local size to use for kernel launch.
   * @param num_bytes_local Number of bytes required per particle.
   * @param stride Number of particles each work item will process, default 1.
   */
  std::vector<ParticleLoopBlockHost> &
  get_range_cell(const std::size_t cell_start, const std::size_t cell_end,
                 std::size_t local_size = 256,
                 const std::size_t num_bytes_local = 0,
                 const std::size_t stride = 1);
};

} // namespace NESO::Particles::ParticleLoopImplementation

#endif
