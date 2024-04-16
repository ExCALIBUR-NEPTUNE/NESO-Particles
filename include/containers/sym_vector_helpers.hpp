#ifndef _NESO_PARTICLES_SYM_VECTOR_HELPERS_H_
#define _NESO_PARTICLES_SYM_VECTOR_HELPERS_H_

#include "../particle_group.hpp"
#include "../particle_sub_group.hpp"
#include "sym_vector.hpp"

namespace NESO::Particles {

/**
 * Helper function to create a SymVector.
 *
 * @param particle_group ParticleGroup to use.
 * @param syms Vector of Syms to use from particle_group.
 */
template <typename T>
SymVectorSharedPtr<T> sym_vector(ParticleGroupSharedPtr particle_group,
                                         std::vector<Sym<T>> syms) {
  return std::make_shared<SymVector<T>>(particle_group, syms);
}

/**
 * Helper function to create a SymVector.
 *
 * @param particle_group ParticleGroup to use.
 * @param syms Syms to use from particle_group.
 */
template <typename T>
SymVectorSharedPtr<T> sym_vector(ParticleGroupSharedPtr particle_group,
                                         std::initializer_list<Sym<T>> syms) {
  return std::make_shared<SymVector<T>>(particle_group, syms);
}

/**
 * Helper function to create a SymVector from a ParticleSubGroup.
 *
 * @param particle_sub_group ParticleSubGroup from which to create SymVector.
 * @param syms Syms to create SymVector from.
 */
template <typename T>
auto sym_vector(ParticleSubGroupSharedPtr particle_sub_group, T syms) {
  return sym_vector(particle_sub_group->get_particle_group(), syms);
}

} // namespace NESO::Particles

#endif
