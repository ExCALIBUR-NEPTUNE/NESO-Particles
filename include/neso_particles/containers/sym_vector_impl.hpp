#ifndef _NESO_PARTICLES_SYM_VECTOR_IMPL_H_
#define _NESO_PARTICLES_SYM_VECTOR_IMPL_H_

#include "../typedefs.hpp"
#include "sym_vector.hpp"
#include "sym_vector_pointer_cache_dispatch.hpp"

namespace NESO::Particles {
namespace ParticleLoopImplementation {

template <typename T>
inline void pre_loop(ParticleLoopGlobalInfo *global_info,
                     Access::Read<SymVector<T> *> &arg) {
  get_sym_vector_cache_dispatch(global_info->particle_group,
                                global_info->particle_sub_group)
      ->create_const(arg.obj->syms);
}
template <typename T>
inline void pre_loop(ParticleLoopGlobalInfo *global_info,
                     Access::Write<SymVector<T> *> &arg) {
  get_sym_vector_cache_dispatch(global_info->particle_group,
                                global_info->particle_sub_group)
      ->create(arg.obj->syms);
}

/**
 * Method to compute access to a SymVector (read).
 */
template <typename T>
inline SymVectorImplGetConstT<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<SymVector<T> *> &a) {
  auto &syms = a.obj->syms;
  return get_sym_vector_cache_dispatch(global_info->particle_group,
                                       global_info->particle_sub_group)
      ->get_const(syms);
}
/**
 * Method to compute access to a SymVector (write).
 */
template <typename T>
inline SymVectorImplGetT<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<SymVector<T> *> &a) {
  auto &syms = a.obj->syms;
  return get_sym_vector_cache_dispatch(global_info->particle_group,
                                       global_info->particle_sub_group)
      ->get(syms);
}

/**
 * Method to compute access to a particle dat (read) - via Sym.
 */
template <typename T>
inline ParticleDatImplGetConstT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<Sym<T> *> &a) {
  auto &sym = *a.obj;
  return get_sym_vector_cache_dispatch(global_info->particle_group,
                                       global_info->particle_sub_group)
      ->get_const(sym);
}
/**
 * Method to compute access to a particle dat (write) - via Sym
 */
template <typename T>
inline ParticleDatImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<Sym<T> *> &a) {
  auto &sym = *a.obj;
  return get_sym_vector_cache_dispatch(global_info->particle_group,
                                       global_info->particle_sub_group)
      ->get(sym);
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
