#ifndef _NESO_PARTICLES_PARTICLE_SUB_GROUP_SELECTOR_SUB_GROUP_UTILITY_HPP_
#define _NESO_PARTICLES_PARTICLE_SUB_GROUP_SELECTOR_SUB_GROUP_UTILITY_HPP_

#include <neso_particles/particle_sub_group/particle_sub_group_base.hpp>

namespace NESO::Particles {

namespace Private {

/**
 * @param particle_group ParticleGroup to return device pointer for number of
 * particles in each cell.
 * @returns Device pointer to number of particles in each cell.
 */
int *get_npart_cell_device_ptr(ParticleGroupSharedPtr particle_group);

/**
 * @param particle_group ParticleGroup to return device pointer for number of
 * particles in each cell.
 * @returns Device pointer to exclusive scan of number of particles in each
 * cell.
 */
INT *get_npart_cell_es_device_ptr(ParticleGroupSharedPtr particle_group);

/**
 * @param particle_sub_group ParticleGroup to return device pointer for number
 * of particles in each cell.
 * @returns Device pointer to number of particles in each cell.
 */
int *get_npart_cell_device_ptr(ParticleSubGroupSharedPtr particle_sub_group);

/**
 * @param particle_sub_group ParticleGroup to return device pointer for number
 * of particles in each cell.
 * @returns Device pointer to exclusive scan of number of particles in each
 * cell.
 */
INT *get_npart_cell_es_device_ptr(ParticleSubGroupSharedPtr particle_sub_group);

} // namespace Private
} // namespace NESO::Particles

#endif
