#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_FUNCTIONS_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_FUNCTIONS_HPP_

#include "../loop/particle_loop_reduction.hpp"
#include "particle_loop_sub_group.hpp"
#include "particle_loop_sub_group_reduction.hpp"

namespace NESO::Particles {

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param name Identifier for particle loop.
 *  @param particle_sub_group ParticleSubGroup to execute kernel for all
 * particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name,
              ParticleSubGroupSharedPtr particle_sub_group, KERNEL kernel,
              ARGS... args) {
  ParticleLoopSharedPtr b;

  if (particle_sub_group->is_entire_particle_group()) {
    if constexpr (Access::HasReduction<ARGS...>::value) {
      b = std::dynamic_pointer_cast<ParticleLoopBase>(
          std::make_shared<ParticleLoopReduction<KERNEL, ARGS...>>(
              name, particle_sub_group->get_particle_group(),
              particle_sub_group, kernel, args...));
    } else {
      b = std::dynamic_pointer_cast<ParticleLoopBase>(
          std::make_shared<ParticleLoop<KERNEL, ARGS...>>(
              name, particle_sub_group->get_particle_group(),
              particle_sub_group, kernel, args...));
    }
  } else {

    if constexpr (Access::HasReduction<ARGS...>::value) {
      b = std::dynamic_pointer_cast<ParticleLoopBase>(
          std::make_shared<ParticleLoopSubGroupReduction<KERNEL, ARGS...>>(
              name, particle_sub_group, kernel, args...));
    } else {
      b = std::dynamic_pointer_cast<ParticleLoopBase>(
          std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
              name, particle_sub_group, kernel, args...));
    }
  }

  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param particle_sub_group ParticleSubGroup to execute kernel for all
 * particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleSubGroupSharedPtr particle_sub_group, KERNEL kernel,
              ARGS... args) {
  return particle_loop("unnamed_kernel", particle_sub_group, kernel, args...);
}

} // namespace NESO::Particles

#endif
