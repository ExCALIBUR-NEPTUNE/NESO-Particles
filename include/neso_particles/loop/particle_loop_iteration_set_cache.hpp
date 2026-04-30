#ifndef _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_CACHE_HPP_
#define _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_CACHE_HPP_

#include "particle_loop_iteration_set.hpp"
#include <map>
#include <memory>

namespace NESO::Particles::ParticleLoopImplementation {

/**
 * This is a container for caching iteration sets for ParticleLoops.
 */
class ParticleLoopIterationSetCache {
protected:
  /// SYCL device
  SYCLTargetSharedPtr sycl_target;
  /// The number of cells.
  const std::size_t ncell;
  /// Host accessible pointer to the number of particles in each cell.
  int *h_npart_cell;
  /// Device accessible pointer to the number of particles in each cell.
  int *d_npart_cell;

  static constexpr std::size_t cache_size_limit = 32;

  // cell_start, cell_end, nbin, local_size, stride
  using cache_key_type = std::tuple<std::size_t, std::size_t, std::size_t,
                                    std::size_t, std::size_t>;
  std::map<cache_key_type, std::unique_ptr<ParticleLoopBlockIterationSet>>
      cache;

public:
  /// Disable (implicit) copies.
  ParticleLoopIterationSetCache(const ParticleLoopIterationSetCache &st) =
      delete;
  /// Disable (implicit) copies.
  ParticleLoopIterationSetCache &
  operator=(ParticleLoopIterationSetCache const &a) = delete;

  /**
   *  Creates iteration set cache.
   *
   *  @param sycl_target Compute device to use.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   *  @param d_npart_cell Device accessible array of cell particle counts.
   */
  ParticleLoopIterationSetCache(SYCLTargetSharedPtr sycl_target,
                                const std::size_t ncell, int *h_npart_cell,
                                int *d_npart_cell);

  /**
   * Empty the cache.
   */
  void clear();

  /**
   * Get an iteration set.
   *
   * @param[in] cell_start First cell.
   * @param[in] cell_end Last cell to visit + 1.
   * @param[in] nbin Default number of bins to use for kernel launch.
   * @param[in] local_size Default local size to use for kernel launch.
   * @param[in] num_bytes_local Number of bytes required per particle.
   * @param[in] stride Number of particles each work item will process,
   * default 1.
   * @param[in, out] iteration_set_size Optionally return the size of the
   * iteration set.
   */
  const std::vector<ParticleLoopBlockHost> *
  get(const std::size_t cell_start, const std::size_t cell_end,
      std::size_t nbin, std::size_t local_size,
      const std::size_t num_bytes_local, const std::size_t stride,
      std::size_t *iteration_set_size = nullptr);
};

} // namespace NESO::Particles::ParticleLoopImplementation

#endif
