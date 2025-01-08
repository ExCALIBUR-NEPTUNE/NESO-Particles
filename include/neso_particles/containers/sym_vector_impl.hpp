#ifndef _NESO_PARTICLES_SYM_VECTOR_IMPL_H_
#define _NESO_PARTICLES_SYM_VECTOR_IMPL_H_

#include "../particle_sub_group/particle_sub_group.hpp"
#include "sym_vector.hpp"

namespace NESO::Particles {

/**
 * Helper function to create a SymVector.
 *
 * @param particle_group ParticleGroup to use.
 * @param syms Vector of Syms to use from particle_group.
 */
template <typename T>
std::shared_ptr<SymVector<T>> sym_vector(ParticleGroupSharedPtr particle_group,
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
std::shared_ptr<SymVector<T>> sym_vector(ParticleGroupSharedPtr particle_group,
                                         std::initializer_list<Sym<T>> syms) {
  return std::make_shared<SymVector<T>>(particle_group, syms);
}

/**
 * Helper function to create a SymVector.
 *
 * @param particle_sub_group ParticleSubGroup to use.
 * @param syms Vector of Syms to use from particle_group.
 */
template <typename T>
std::shared_ptr<SymVector<T>>
sym_vector(ParticleSubGroupSharedPtr particle_sub_group,
           std::vector<Sym<T>> syms) {
  return sym_vector(particle_sub_group->get_particle_group(), syms);
}

/**
 * Helper function to create a SymVector.
 *
 * @param particle_sub_group ParticleSubGroup to use.
 * @param syms Syms to use from particle_group.
 */
template <typename T>
std::shared_ptr<SymVector<T>>
sym_vector(ParticleSubGroupSharedPtr particle_sub_group,
           std::initializer_list<Sym<T>> syms) {
  return sym_vector(particle_sub_group->get_particle_group(), syms);
}

} // namespace NESO::Particles

#endif
