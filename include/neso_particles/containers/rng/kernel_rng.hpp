#ifndef _NESO_PARTICLES_KERNEL_RNG_H_
#define _NESO_PARTICLES_KERNEL_RNG_H_

#include "../../loop/particle_loop_base.hpp"
#include "../../loop/particle_loop_index.hpp"
#include "../../particle_group.hpp"

#include <functional>
#include <tuple>
#include <type_traits>

namespace NESO::Particles {

template <typename T> struct KernelRNG;

namespace Access::KernelRNG {

/**
 * This is the kernel type for KernelRNG which is used for all
 * implementations which present RNG values to the kernel via an allocated
 * device buffer.
 */
template <typename T> struct Read {
  /**
   * Device data for the device RNG.
   */
  T data;

  /**
   * Access the RNG data for this particle.
   *
   * @param particle_index Particle index to access.
   * @param component RNG component to access.
   * @returns Constant reference to RNG data.
   */
  inline auto at(const Access::LoopIndex::Read &particle_index,
                 const int component) {
    return this->data.at(particle_index, component);
  }
};

} // namespace Access::KernelRNG

namespace ParticleLoopImplementation {
/**
 * The templates in this section match types which are derived from
 * KernelRNG<T> for some template parameter T.
 */

/**
 * LoopParameter type for a KernelRNG.
 */
template <typename T>
struct LoopParameter<
    Access::Read<T>,
    typename std::enable_if<
        std::is_base_of<KernelRNG<typename T::SpecialisationType>, T>::value,
        std::true_type>::type> {
  using type = Access::KernelRNG::Read<typename T::SpecialisationType>;
};

/**
 * Create the loop argument for a KernelRNG.
 */
template <
    typename T,
    std::enable_if_t<
        std::is_base_of<KernelRNG<typename T::SpecialisationType>, T>::value,
        bool> = true>
inline Access::KernelRNG::Read<typename T::SpecialisationType>
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                sycl::handler &cgh, Access::Read<T *> &a) {
  return a.obj->impl_get_const(global_info);
}

/**
 * Create the kernel argument for a KernelRNG.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              Access::KernelRNG::Read<T> &rhs,
                              Access::KernelRNG::Read<T> &lhs) {
  lhs = rhs;
}

/**
 *  KernelParameter type for read-only access to a KernelRNG.
 */
template <typename T>
struct KernelParameter<
    Access::Read<T>,
    typename std::enable_if<
        std::is_base_of<KernelRNG<typename T::SpecialisationType>, T>::value,
        std::true_type>::type> {
  using type = Access::KernelRNG::Read<typename T::SpecialisationType>;
};

/**
 * The function called before the ParticleLoop is executed.
 */
template <
    typename T,
    std::enable_if_t<
        std::is_base_of<KernelRNG<typename T::SpecialisationType>, T>::value,
        bool> = true>
inline void pre_loop(ParticleLoopGlobalInfo *global_info,
                     Access::Read<T *> &arg) {
  arg.obj->impl_pre_loop_read(global_info);
}

/**
 * The function called after the ParticleLoop is executed.
 */
template <
    typename T,
    std::enable_if_t<
        std::is_base_of<KernelRNG<typename T::SpecialisationType>, T>::value,
        bool> = true>
inline void post_loop(ParticleLoopGlobalInfo *global_info,
                      Access::Read<T *> &arg) {
  arg.obj->impl_post_loop_read(global_info);
}

} // namespace ParticleLoopImplementation

namespace {

template <typename T> struct get_rng_value_type;

template <template <typename> typename T, typename U>
struct get_rng_value_type<T<U>> {
  using type = U;
};

} // namespace

/**
 * Abstract base class for RNG implementations which create a block of random
 * numbers on the device prior to the loop execution.
 */
template <typename T> struct KernelRNG {

  /**
   * This type stores the templated type such that we can retrive it on derived
   * types and test if a type is a derived type of KernelRNG<T> for some
   * type T. The derived types will inherit SpecialisationType as a member
   * type.
   */
  using SpecialisationType = T;

  /**
   * This type is the type of the parameter that should be used in a
   * ParticeLoop kernel.
   */
  using KernelType = Access::KernelRNG::Read<T>;

  /**
   * This type is the, probably scalar, type that the RNG returns when sampled.
   */
  using RNGValueType = typename get_rng_value_type<T>::type;

  /**
   * Create the loop arguments for the RNG implementation.
   *
   * @param global_info Global information for the loop about to be executed.
   * @returns Pointer to the device data that contains the RNG data.
   */
  virtual inline Access::KernelRNG::Read<T> impl_get_const(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) = 0;

  /**
   * Executed by the loop pre execution.
   * @param global_info Global information for the loop which is to be executed.
   */
  virtual inline void impl_pre_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) = 0;

  /**
   * Executed by the loop post execution.
   * @param global_info Global information for the loop which completed.
   */
  virtual inline void impl_post_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) = 0;

  /**
   * @returns True if no errors have been detected otherwise false.
   */
  virtual inline bool valid_internal_state() = 0;
};

} // namespace NESO::Particles

#endif
