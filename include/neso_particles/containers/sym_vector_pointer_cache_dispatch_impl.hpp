#ifndef _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_DISPATCH_IMPL_HPP_
#define _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_DISPATCH_IMPL_HPP_

#include "../particle_sub_group/particle_sub_group.hpp"
#include "sym_vector_pointer_cache_dispatch.hpp"

namespace NESO::Particles {
inline SymVectorPointerCacheDispatchSharedPtr
get_sym_vector_cache_dispatch(ParticleGroup *particle_group,
                              ParticleSubGroup *particle_sub_group) {
  if (particle_sub_group != nullptr) {
    return particle_sub_group->ephemeral.sym_vector_pointer_cache_dispatch;
  } else {
    return particle_group->sym_vector_pointer_cache_dispatch;
  }
}

} // namespace NESO::Particles
#endif
