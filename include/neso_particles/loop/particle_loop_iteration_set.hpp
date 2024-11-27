#ifndef _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_
#define _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_

#include "../compute_target.hpp"
#include "../particle_dat.hpp"
#include "../sycl_typedefs.hpp"
#include "../typedefs.hpp"

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
  ParticleLoopIterationSet(const int nbin, const int ncell, int *h_npart_cell)
      : nbin(std::min(ncell, nbin)), ncell(ncell), h_npart_cell(h_npart_cell) {
    this->iteration_set.reserve(nbin);
    this->cell_offsets.reserve(nbin);
  }

  /**
   *  Create and return an iteration set which is formed as nbin
   *  sycl::nd_ranges.
   *
   *  @param cell If set iteration set will only cover this cell.
   *  @param local_size Optional size of SYCL work groups.
   *  @returns Tuple containing: Number of bins, sycl::nd_ranges, cell index
   *  offsets.
   */
  inline std::tuple<int, std::vector<sycl::nd_range<2>> &,
                    std::vector<std::size_t> &>
  get(const std::optional<int> cell = std::nullopt,
      const size_t local_size = 256) {

    this->iteration_set.clear();
    this->cell_offsets.clear();
    this->iteration_set_size = 0;

    if (cell == std::nullopt) {
      for (int binx = 0; binx < nbin; binx++) {
        int start, end;
        get_decomp_1d(nbin, ncell, binx, &start, &end);
        const int bin_width = end - start;
        int cell_maxi = 0;
        int cell_avg = 0;
        for (int cellx = start; cellx < end; cellx++) {
          const int cell_occ = h_npart_cell[cellx];
          cell_maxi = std::max(cell_maxi, cell_occ);
          cell_avg += cell_occ;
          this->iteration_set_size += cell_occ;
        }
        cell_avg = (((REAL)cell_avg) / ((REAL)(end - start)));
        const size_t cell_local_size =
            get_min_power_of_two((size_t)cell_avg, local_size);
        const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                      static_cast<long long>(cell_local_size));
        const std::size_t outer_size =
            static_cast<std::size_t>(div_mod.quot +
                                     (div_mod.rem == 0 ? 0 : 1)) *
            cell_local_size;

        if (cell_maxi > 0) {
          this->iteration_set.emplace_back(
              sycl::nd_range<2>(sycl::range<2>(bin_width, outer_size),
                                sycl::range<2>(1, cell_local_size)));
          this->cell_offsets.push_back(static_cast<std::size_t>(start));
        }
      }

      return {this->iteration_set.size(), this->iteration_set,
              this->cell_offsets};
    } else {
      const int cellx = cell.value();
      const size_t cell_maxi = static_cast<size_t>(h_npart_cell[cellx]);

      this->iteration_set_size = cell_maxi;
      const size_t cell_local_size =
          get_min_power_of_two((size_t)cell_maxi, local_size);

      const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                    static_cast<long long>(cell_local_size));
      const std::size_t outer_size =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1)) *
          cell_local_size;
      this->iteration_set.emplace_back(sycl::nd_range<2>(
          sycl::range<2>(1, outer_size), sycl::range<2>(1, cell_local_size)));
      this->cell_offsets.push_back(static_cast<std::size_t>(cellx));
      return {1, this->iteration_set, this->cell_offsets};
    }
  }
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

  inline std::size_t get_global_size(const std::size_t N,
                                     const std::size_t local_size) {
    const auto div_mod =
        std::div(static_cast<long long>(N), static_cast<long long>(local_size));
    const std::size_t outer_size =
        static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1)) *
        local_size;
    return outer_size;
  }

  inline std::size_t
  get_local_size(std::size_t local_size,
                 const std::size_t num_bytes_local, // this is per particle
                 const std::size_t stride) {
    const std::size_t local_mem_size =
        this->sycl_target->device_limits.local_mem_size;
    const std::size_t num_bytes_per_block = stride * num_bytes_local;
    NESOASSERT(num_bytes_per_block <= local_mem_size,
               "Impossible to create a local range for this stride and local "
               "memory size.");
    const std::size_t max_num_blocks_per_workgroup =
        (num_bytes_local == 0) ? local_size
                               : local_mem_size / num_bytes_per_block;
    local_size = std::min(get_prev_power_of_two(max_num_blocks_per_workgroup), local_size);
    NESOASSERT(local_size * stride * num_bytes_local <= local_mem_size,
               "Failure to determine a local size for iteration set.");
    return local_size;
  }

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
                                int *d_npart_cell)
      : sycl_target(sycl_target), ncell(ncell), h_npart_cell(h_npart_cell),
        d_npart_cell(d_npart_cell) {}

  /**
   *  Creates iteration set creator for a given ParticleDat.
   *
   *  @param particle_dat Specify an iteration set from a ParticleDat.
   */
  template <typename T>
  ParticleLoopBlockIterationSet(ParticleDatSharedPtr<T> particle_dat)
      : sycl_target(particle_dat->sycl_target), ncell(particle_dat->ncell),
        h_npart_cell(particle_dat->h_npart_cell),
        d_npart_cell(particle_dat->d_npart_cell) {}

  /**
   * Get a complete iteration set for a particle loop for all cells.
   *
   * @param nbin Default number of bins to use for kernel launch.
   * @param local_size Default local size to use for kernel launch.
   * @param num_bytes_local Number of bytes required per particle.
   * @param stride Number of particles each work item will process, default 1.
   */
  inline std::vector<ParticleLoopBlockHost> &
  get_all_cells(std::size_t nbin = 16, std::size_t local_size = 256,
                const std::size_t num_bytes_local = 0,
                const std::size_t stride = 1) {

    local_size = this->get_local_size(local_size, num_bytes_local, stride);
    nbin = std::min(nbin, this->ncell);
    this->iteration_set.clear();

    // Create an iteration block that covers all cells up to a multiple of the
    // local size.
    std::size_t min_occupancy = std::numeric_limits<std::size_t>::max();
    for (std::size_t cellx = 0; cellx < this->ncell; cellx++) {
      min_occupancy = std::min(
          min_occupancy, static_cast<std::size_t>(this->h_npart_cell[cellx]));
    }
    // Truncate to a multiple of local size and stride.
    min_occupancy /= (local_size * stride);
    min_occupancy *= (local_size * stride);
    this->iteration_set_size = min_occupancy * this->ncell;
    if (min_occupancy > 0) {
      ParticleLoopBlockDevice block_device{0, 0, this->d_npart_cell, stride};
      this->iteration_set.emplace_back(
          block_device, false, local_size,
          this->sycl_target->device_limits.validate_nd_range(sycl::nd_range<2>(
              // min_occupancy is already a multiple of local_size and stride by
              // construction.
              sycl::range<2>(this->ncell, min_occupancy / stride),
              sycl::range<2>(1, local_size))));
    }
    // Create the peel loops
    for (std::size_t binx = 0; binx < nbin; binx++) {
      std::size_t start, end;
      get_decomp_1d(nbin, ncell, binx, &start, &end);
      const std::size_t bin_width = end - start;
      std::size_t cell_max_occ = 0;
      for (std::size_t cellx = start; cellx < end; cellx++) {
        const std::size_t npart =
            static_cast<std::size_t>(this->h_npart_cell[cellx]);
        this->iteration_set_size += npart;
        cell_max_occ = std::max(cell_max_occ, npart);
      }
      // Subtract of the block already completed as this is a peel loop.
      cell_max_occ -= min_occupancy;
      if (cell_max_occ > 0) {
        const std::size_t global_range = this->get_global_size(
            get_next_multiple(cell_max_occ, stride) / stride, local_size);
        ParticleLoopBlockDevice block_device{start, min_occupancy / stride,
                                             this->d_npart_cell, stride};
        this->iteration_set.emplace_back(
            block_device, true, local_size,
            this->sycl_target->device_limits.validate_nd_range(
                sycl::nd_range<2>(sycl::range<2>(bin_width, global_range),
                                  sycl::range<2>(1, local_size))));
      }
    }

    return this->iteration_set;
  }

  /**
   * Get an iteration set for a particle loop for a single cell.
   *
   * @param cell Produce an iteration set for a single cell.
   * @param local_size Default local size to use for kernel launch.
   * @param num_bytes_local Number of bytes required per particle.
   * @param stride Number of particles each work item will process, default 1.
   */
  inline std::vector<ParticleLoopBlockHost> &
  get_single_cell(const std::size_t cell, std::size_t local_size = 256,
                  const std::size_t num_bytes_local = 0,
                  const std::size_t stride = 1) {

    local_size = this->get_local_size(local_size, num_bytes_local, stride);
    this->iteration_set.clear();

    const std::size_t npart =
        static_cast<std::size_t>(this->h_npart_cell[cell]);
    const std::size_t global_range = this->get_global_size(
        get_next_multiple(npart, stride) / stride, local_size);

    ParticleLoopBlockDevice block_device{cell, 0, this->d_npart_cell, stride};

    this->iteration_set.emplace_back(
        block_device, true, local_size,
        this->sycl_target->device_limits.validate_nd_range(sycl::nd_range<2>(
            sycl::range<2>(1, global_range), sycl::range<2>(1, local_size))));

    this->iteration_set_size = npart;
    return this->iteration_set;
  }
};

} // namespace NESO::Particles::ParticleLoopImplementation

#endif
