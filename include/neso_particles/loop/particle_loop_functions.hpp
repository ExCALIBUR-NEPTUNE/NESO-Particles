#ifndef _NESO_PARTICLES_LOOP_PARTICLE_LOOP_FUNCTIONS_H_
#define _NESO_PARTICLES_LOOP_PARTICLE_LOOP_FUNCTIONS_H_

#include <cstdlib>
#include <string>

#include "particle_loop.hpp"
#include "particle_loop_reduction.hpp"

namespace NESO::Particles {

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleGroup.
 *
 *  @param name Identifier for particle loop.
 *  @param particle_group ParticleGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 *              type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name, ParticleGroupSharedPtr particle_group,
              KERNEL kernel, ARGS... args) {
  if constexpr (Access::HasReduction<ARGS...>::value) {
    auto p = std::make_shared<ParticleLoopReduction<KERNEL, ARGS...>>(
        name, particle_group, kernel, args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  } else {
    auto p = std::make_shared<ParticleLoop<KERNEL, ARGS...>>(
        name, particle_group, kernel, args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  }
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleGroup.
 *
 *  @param particle_group ParticleGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 *              type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleGroupSharedPtr particle_group, KERNEL kernel,
              ARGS... args) {
  return particle_loop("unnamed_kernel", particle_group, kernel, args...);
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleDat.
 *
 *  @param name Identifier for particle loop.
 *  @param particle_dat ParticleDat to define the iteration set.
 *  @param kernel Kernel to execute for all particles in the ParticleGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename DAT_TYPE, typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name,
              ParticleDatSharedPtr<DAT_TYPE> particle_dat, KERNEL kernel,
              ARGS... args) {

  static_assert(!Access::HasReduction<ARGS...>::value,
                "Reductions access descriptors not supported with ParticleDat "
                "iteration sets.");

  auto p = std::make_shared<ParticleLoop<KERNEL, ARGS...>>(name, particle_dat,
                                                           kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleDat.
 *
 *  @param particle_dat ParticleDat to define the iteration set.
 *  @param kernel Kernel to execute for all particles in the ParticleGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename DAT_TYPE, typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleDatSharedPtr<DAT_TYPE> particle_dat, KERNEL kernel,
              ARGS... args) {
  return particle_loop("unnamed_kernel", particle_dat, kernel, args...);
}

} // namespace NESO::Particles

#endif
