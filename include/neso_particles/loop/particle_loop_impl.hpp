#ifndef _NESO_PARTICLES_LOOP_PARTICLE_LOOP_IMPL_HPP_
#define _NESO_PARTICLES_LOOP_PARTICLE_LOOP_IMPL_HPP_

#include "../containers/sym_vector_impl.hpp"
#include "particle_loop.hpp"

namespace NESO::Particles {

/**
 * Method to compute access to a type wrapped in a shared_ptr.
 */
template <typename KERNEL, typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoop<KERNEL, ARGS...>::pre_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    T<std::shared_ptr<U>> a) {
  T<U *> c = {a.obj.get()};
  ParticleLoopImplementation::pre_loop(global_info, c);
}

/**
 * Method to compute access to a type not wrapper in a shared_ptr
 */
template <typename KERNEL, typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoop<KERNEL, ARGS...>::pre_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info, T<U> a) {
  T<U *> c = {&a.obj};
  ParticleLoopImplementation::pre_loop(global_info, c);
}

/**
 * Method to compute access to a type wrapped in a shared_ptr.
 */
template <typename KERNEL, typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoop<KERNEL, ARGS...>::create_loop_arg_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    sycl::handler &cgh, T<std::shared_ptr<U>> a) {
  T<U *> c = {a.obj.get()};
  return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
}
/**
 * Method to compute access to a type not wrapper in a shared_ptr
 */

template <typename KERNEL, typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoop<KERNEL, ARGS...>::create_loop_arg_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    sycl::handler &cgh, T<U> a) {
  T<U *> c = {&a.obj};
  return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
}

/**
 * Method to compute access to a type wrapped in a shared_ptr.
 */

template <typename KERNEL, typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoop<KERNEL, ARGS...>::post_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    T<std::shared_ptr<U>> a) {
  T<U *> c = {a.obj.get()};
  ParticleLoopImplementation::post_loop(global_info, c);
}
/**
 * Method to compute access to a type not wrapper in a shared_ptr
 */

template <typename KERNEL, typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoop<KERNEL, ARGS...>::post_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info, T<U> a) {
  T<U *> c = {&a.obj};
  ParticleLoopImplementation::post_loop(global_info, c);
}

} // namespace NESO::Particles

#endif
