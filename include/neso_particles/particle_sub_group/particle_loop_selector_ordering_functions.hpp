#ifndef _NESO_PARTICLES_LOOP_PARTICLE_LOOP_SELECTOR_ORDERING_FUNCTIONS_H_
#define _NESO_PARTICLES_LOOP_PARTICLE_LOOP_SELECTOR_ORDERING_FUNCTIONS_H_

#include "particle_loop_selector_ordering.hpp"

namespace NESO::Particles::Private {

template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop_selector_ordering(int **hd_cell_counts, int **hd_layers,
                                ParticleGroupSharedPtr particle_group,
                                KERNEL kernel, ARGS... args) {
  auto p = std::make_shared<ParticleLoopSelectorOrdering<KERNEL, ARGS...>>(
      hd_cell_counts, hd_layers, particle_group, kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

} // namespace NESO::Particles::Private
#endif
