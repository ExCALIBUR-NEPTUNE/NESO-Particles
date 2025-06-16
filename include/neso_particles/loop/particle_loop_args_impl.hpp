#ifndef _NESO_PARTICLES_LOOP_PARTICLE_LOOP_ARGS_IMPL_HPP_
#define _NESO_PARTICLES_LOOP_PARTICLE_LOOP_ARGS_IMPL_HPP_

#include "particle_loop_args.hpp"

namespace NESO::Particles {

/**
 * Method to compute access to a type wrapped in a shared_ptr.
 */
template <typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoopArgs<ARGS...>::create_loop_arg_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    sycl::handler &cgh, T<std::shared_ptr<U>> a) {
  T<U *> c = {a.obj.get()};
  return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
}
/**
 * Method to compute access to a type not wrapper in a shared_ptr
 */

template <typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoopArgs<ARGS...>::create_loop_arg_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    sycl::handler &cgh, T<U> a) {
  T<U *> c = {&a.obj};
  return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
}

/**
 * Method to compute access to a Reduction type.
 */

template <typename... ARGS>
template <template <typename> typename T, typename U, typename OP>
inline auto ParticleLoopArgs<ARGS...>::create_loop_arg_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    sycl::handler &cgh, Access::Reduction<std::shared_ptr<T<U>>, OP> a) {
  return ParticleLoopImplementation::create_loop_arg(global_info, cgh, a);
}

/**
 * Method to compute access to a type wrapped in a shared_ptr.
 */
template <typename... ARGS>
template <template <typename> typename T, typename U>
inline void ParticleLoopArgs<ARGS...>::pre_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    T<std::shared_ptr<U>> a) {
  T<U *> c = {a.obj.get()};
  ParticleLoopImplementation::pre_loop(global_info, c);
}

/**
 * Method to compute access to a type not wrapper in a shared_ptr
 */
template <typename... ARGS>
template <template <typename> typename T, typename U>
inline void ParticleLoopArgs<ARGS...>::pre_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info, T<U> a) {
  T<U *> c = {&a.obj};
  ParticleLoopImplementation::pre_loop(global_info, c);
}

/**
 * Pre loop cast for reduction access
 */
template <typename... ARGS>
template <template <typename> typename T, typename U, typename OP>
inline void ParticleLoopArgs<ARGS...>::pre_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    Access::Reduction<std::shared_ptr<T<U>>, OP> a) {
  ParticleLoopImplementation::pre_loop(global_info, a);
}

/**
 * Method to compute access to a type wrapped in a shared_ptr.
 */
template <typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoopArgs<ARGS...>::post_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    T<std::shared_ptr<U>> a) {
  T<U *> c = {a.obj.get()};
  ParticleLoopImplementation::post_loop(global_info, c);
}
/**
 * Method to compute access to a type not wrapper in a shared_ptr
 */

template <typename... ARGS>
template <template <typename> typename T, typename U>
inline auto ParticleLoopArgs<ARGS...>::post_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info, T<U> a) {
  T<U *> c = {&a.obj};
  ParticleLoopImplementation::post_loop(global_info, c);
}

/**
 * Post loop cast for reduction access
 */
template <typename... ARGS>
template <template <typename> typename T, typename U, typename OP>
inline void ParticleLoopArgs<ARGS...>::post_loop_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    Access::Reduction<std::shared_ptr<T<U>>, OP> a) {
  ParticleLoopImplementation::post_loop(global_info, a);
}

} // namespace NESO::Particles

#endif
