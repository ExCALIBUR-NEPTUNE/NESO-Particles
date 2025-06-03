#ifndef _NESO_PARTICLES_PARTICLE_SUB_GROUP_PARTICLE_SUB_GROUP_UTILITY_H_
#define _NESO_PARTICLES_PARTICLE_SUB_GROUP_PARTICLE_SUB_GROUP_UTILITY_H_
#include "../particle_group.hpp"
#include "particle_sub_group_base.hpp"

namespace NESO::Particles {

/**
 * Helper function to return the underlying ParticleGroup for a type.
 *
 * @param particle_sub_group ParticleSubGroup.
 * @returns Underlying ParticleGroup.
 */
inline auto get_particle_group(ParticleSubGroupSharedPtr particle_sub_group) {
  return particle_sub_group->get_particle_group();
}

/**
 * Helper function to return the underlying ParticleGroup for a type.
 *
 * @param particle_group ParticleGroup.
 * @returns Underlying ParticleGroup.
 */
inline auto get_particle_group(ParticleGroupSharedPtr particle_group) {
  return particle_group;
}

/**
 * Helper function for determining if a templated type is a ParticleGroup or
 * ParticleSubGroup.
 *
 * @param p ParticleGroupSharedPtr.
 * @returns False.
 */
constexpr inline bool
is_particle_sub_group([[maybe_unused]] ParticleGroupSharedPtr &p) {
  return false;
}

/**
 * Helper function for determining if a templated type is a ParticleGroup or
 * ParticleSubGroup.
 *
 * @param p ParticleSubGroupSharedPtr.
 * @returns True.
 */
constexpr inline bool
is_particle_sub_group([[maybe_unused]] ParticleSubGroupSharedPtr &p) {
  return true;
}

} // namespace NESO::Particles

#endif
