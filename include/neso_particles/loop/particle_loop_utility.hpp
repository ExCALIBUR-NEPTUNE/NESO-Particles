#ifndef _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#include "particle_loop_base.hpp"

namespace NESO::Particles {

/**
 * This function will inspect the particle group and particle sub group part of
 * the metadata. For many use cases get_loop_iteration_set_size may be more
 * appropriate.
 *
 * @param global_info Metadata for loop launch.
 * @returns The number of particles in the iteration set.
 */
std::size_t
get_loop_npart(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info);

/**
 *
 * This function will inspect the particle group and particle sub group part of
 * the metadata if there is no explicitly set loop size. This function will also
 * update the iteration set size on the global info to be the computed number of
 * particles in the loop in the scenario where no loop size was provided.
 *
 * @param[in, out] global_info Metadata for loop launch.
 * @returns The size of the iteration set specified by the metadata.
 */
std::size_t get_loop_iteration_set_size(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info);

} // namespace NESO::Particles
#endif
