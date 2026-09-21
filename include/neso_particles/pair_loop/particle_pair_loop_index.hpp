#ifndef _NESO_PARTICLES_PAIR_LOOP_PARTICLE_PAIR_LOOP_INDEX_HPP_
#define _NESO_PARTICLES_PAIR_LOOP_PARTICLE_PAIR_LOOP_INDEX_HPP_

#include "particle_pair_loop_base.hpp"

/**
 * @defgroup particle_pair_loop_particle_pair_loop_index ParticlePairLoopIndex
 * @ingroup particle_pair_loop
 * @details Particle Pair Loop construct for identifying the indices of
 * particles in pairs.
 */

namespace NESO::Particles {

/**
 * @ingroup particle_pair_loop_particle_pair_loop_index
 *
 * Pass to pair loops with an access descriptor for the corresponding particle:
 * ```
 * Access::A(Access::read(ParticlePairLoopIndex{})),
 * Access::B(Access::read(ParticlePairLoopIndex{})),
 * ...
 * ```
 */
struct ParticlePairLoopIndex {};

namespace Access::PairLoopIndex {

/**
 * Kernel type for read-only access to a ParticlePairLoopIndex. When the access
 * descriptor is for particle A then the kernel argument will describe particle
 * A. When the access descriptor is for particle B then the kernel argument will
 * describe particle B.
 *
 * @ingroup particle_pair_loop_particle_pair_loop_index
 */
struct Read {
  INT linear_index{-1};
  INT loop_linear_index{-1};

  /// The cell containing the particle. Use Access::A and Access::B to
  /// distinguish between a and b.
  INT cell{-1};

  /// The layer of the particle. Use Access::A and Access::B to distinguish
  /// between a and b.
  INT layer{-1};

  /**
   * @returns The linear index of the pair in the loop.
   */
  inline INT get_loop_linear_index() const { return loop_linear_index; }

  /**
   * @returns The linear index of the pair in the pair list.
   */
  inline INT get_local_linear_index() const { return linear_index; }
};

} // namespace Access::PairLoopIndex

namespace ParticleLoopImplementation {

struct ParticlePairLoopIndexKernelT {};

/**
 *  KernelParameter type for read-only access to a ParticlePairLoopIndex.
 */
template <> struct KernelParameter<Access::Read<ParticlePairLoopIndex>> {
  using type = Access::PairLoopIndex::Read;
};

/**
 *  Loop parameter for read access of a ParticlePairLoopIndex.
 */
template <> struct LoopParameter<Access::Read<ParticlePairLoopIndex>> {
  using type = ParticlePairLoopIndexKernelT;
};

/**
 * Method to compute access to a ParticlePairLoopIndex (read)
 */
inline ParticlePairLoopIndexKernelT create_loop_arg(
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopGlobalInfo
        *global_info,
    [[maybe_unused]] sycl::handler &cgh,
    [[maybe_unused]] Access::Read<ParticlePairLoopIndex *> &a) {
  return {};
}

} // namespace ParticleLoopImplementation

namespace ParticlePairLoopImplementation {

/**
 * Function to create the kernel argument for ParticlePairLoopIndex read
 * access.
 */
inline void create_kernel_arg(
    ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    [[maybe_unused]] ParticleLoopImplementation::ParticlePairLoopIndexKernelT
        &rhs,
    Access::PairLoopIndex::Read &lhs) {

  lhs.linear_index = iteration.pair_index;
  lhs.loop_linear_index = iteration.loop_pair_index;
  lhs.cell = iteration_particle.cellx;
  lhs.layer = iteration_particle.layerx;
}
} // namespace ParticlePairLoopImplementation
} // namespace NESO::Particles

#endif
