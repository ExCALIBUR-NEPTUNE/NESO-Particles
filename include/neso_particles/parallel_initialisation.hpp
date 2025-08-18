#ifndef __NESO_PARTICLES_PARALLEL_INITIALISATION
#define __NESO_PARTICLES_PARALLEL_INITIALISATION

#include "particle_group.hpp"

namespace NESO::Particles {

/**
 * Function used by parallel_advection_initialisation to step particles along a
 * line to destination positions. Gets a safe temporary position in the local
 * subdomain from which to start particle movement.
 *
 * @param particle_group ParticleGroup being initialised.
 * @param point Output point in local subdomain.
 */
void get_point_in_local_domain(ParticleGroupSharedPtr particle_group,
                               double *point);

/**
 *  Function used by parallel_advection_initialisation to step particles along a
 * line to destination positions. Stores the original particle positions.
 *
 *  @param particle_group ParticleGroup being initialised.
 */
void parallel_advection_store(ParticleGroupSharedPtr particle_group);

/**
 *  Function used by parallel_advection_initialisation to step particles along a
 * line to destination positions. Sets the particle positions to the original
 * positions specified by the user.
 *
 *  @param particle_group ParticleGroup being initialised.
 */
void parallel_advection_restore(ParticleGroupSharedPtr particle_group);

/**
 *  Function used by parallel_advection_initialisation to step particles along a
 *  line to the original positions specified by the user.
 *
 *  @param particle_group ParticleGroup being initialised.
 *  @param num_steps Number of steps over which the stepping occurs.
 *  @param step The current step out of num_steps.
 */
void parallel_advection_step(ParticleGroupSharedPtr particle_group,
                             const int num_steps, const int step);

/**
 *  Initialisation utility to aid parallel creation of particle distributions.
 *  This function performs the all-to-all movement that occurs when a
 *  ParticleGroup contains particles with positions (far) outside the owned
 *  region of space. For example consider if each MPI rank creates N particles
 *  uniformly distributed over the entire simulation domain then the call to
 *  `hybrid_move` would have a cost equal to the number of MPI ranks squared.
 *
 *  This function gives each particle a temporary position in the owned
 *  subdomain then moves the particles in a straight line to the original
 *  positions in the positions ParticleDat. It is assumed that the simulation
 *  domain is convex.
 *
 *  This function is used by adding particles with
 *  `ParticleGroup.add_particles_local` on each rank then collectively calling
 *  this function.
 *
 *  @param particle_group ParticleGroup to initialise by moving particles to the
 * positions in the position ParticleDat.
 *  @param num_steps optional number of steps to move particles over.
 */
void parallel_advection_initialisation(ParticleGroupSharedPtr particle_group,
                                       const int num_steps = 20);

} // namespace NESO::Particles

#endif
