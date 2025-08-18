#ifndef _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#include "particle_loop_base.hpp"

namespace NESO::Particles {

/**
 * @returns The number of particles in the iteration set.
 */
int get_loop_npart(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info);

} // namespace NESO::Particles
#endif
