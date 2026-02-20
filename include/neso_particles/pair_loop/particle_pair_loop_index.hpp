#ifndef _NESO_PARTICLES_PAIR_LOOP_PARTICLE_PAIR_LOOP_INDEX_HPP_
#define _NESO_PARTICLES_PAIR_LOOP_PARTICLE_PAIR_LOOP_INDEX_HPP_

#include "particle_pair_loop_base.hpp"

namespace NESO::Particles {

struct ParticlePairLoopIndex {};

namespace Access::PairLoopIndex {

/**
 *  Kernel type for read-only access to a ParticlePairLoopIndex.
 */
struct Read {
  INT linear_index{-1};

  /**
   * @returns The linear index of the pair in the loop.
   */
  inline INT get_loop_linear_index() const { return linear_index; }
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
 *  Function to create the kernel argument for ParticlePairLoopIndex read
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
}
} // namespace ParticlePairLoopImplementation
} // namespace NESO::Particles

#endif
