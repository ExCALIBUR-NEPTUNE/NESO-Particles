#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_FUNCTIONS_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_FUNCTIONS_HPP_

#include "particle_loop_sub_group.hpp"
#include "particle_loop_sub_group_reduction.hpp"

namespace NESO::Particles {

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param name Identifier for particle loop.
 *  @param particle_group ParticleSubGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name, ParticleSubGroupSharedPtr particle_group,
              KERNEL kernel, ARGS... args) {
  if (particle_group->is_entire_particle_group()) {
    auto p = std::make_shared<ParticleLoop<KERNEL, ARGS...>>(
        name, particle_group->get_particle_group(), particle_group, kernel,
        args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  } else {
    auto p = std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
        name, particle_group, kernel, args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  }
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param particle_group ParticleSubGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleSubGroupSharedPtr particle_group, KERNEL kernel,
              ARGS... args) {
  return particle_loop("unnamed_kernel", particle_group, kernel, args...);
}

} // namespace NESO::Particles

#endif
