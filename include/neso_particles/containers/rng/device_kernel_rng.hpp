#ifndef _NESO_PARTICLES_DEVICE_KERNEL_RNG_H_
#define _NESO_PARTICLES_DEVICE_KERNEL_RNG_H_

#include "../../loop/particle_loop_base.hpp"
#include "../../loop/particle_loop_index.hpp"
#include "../../particle_group.hpp"

#include <functional>
#include <tuple>
#include <type_traits>

namespace NESO::Particles {

template <typename T> struct DeviceKernelRNG;

namespace Access::DeviceKernelRNG {

/**
 * This is the kernel type for DeviceKernelRNG which is used for all
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

} // namespace Access::DeviceKernelRNG

namespace ParticleLoopImplementation {

///**
// * LoopParameter type for a DeviceKernelRNG.
// */
//template <typename T> struct LoopParameter<Access::Read<DeviceKernelRNG<T>>> {
//  using type = Access::DeviceKernelRNG::Read<T>;
//};
//
/**
 * Create the loop argument for a DeviceKernelRNG.
 */
// template <typename T>
// inline Access::DeviceKernelRNG::Read<T>
// create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
//                 sycl::handler &cgh, Access::Read<DeviceKernelRNG<T> *> &a) {
//   return a.obj->impl_get_const(global_info);
// }



/**
 * LoopParameter type for a DeviceKernelRNG.
 */
template<typename T>
struct LoopParameter<
        Access::Read<T>, 
        typename std::enable_if<
            std::is_base_of<DeviceKernelRNG<typename T::KernelSpecialisation>, T>::value, 
            std::true_type
        >::type
    > {
    using type = Access::DeviceKernelRNG::Read<typename T::KernelSpecialisation>;
};


template<
    typename T, 
    std::enable_if_t<
        std::is_base_of<DeviceKernelRNG<typename T::KernelSpecialisation>, T>::value, bool
    > = true
>
inline Access::DeviceKernelRNG::Read<typename T::KernelSpecialisation>
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                sycl::handler &cgh, Access::Read<T *> &a) {
  return a.obj->impl_get_const(global_info);
}


/**
 * Create the kernel argument for a DeviceKernelRNG.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              Access::DeviceKernelRNG::Read<T> &rhs,
                              Access::DeviceKernelRNG::Read<T> &lhs) {
  lhs = rhs;
}

template<typename T>
struct KernelParameter<
        Access::Read<T>, 
        typename std::enable_if<
            std::is_base_of<DeviceKernelRNG<typename T::KernelSpecialisation>, T>::value, 
            std::true_type
        >::type
    > {
    using type = Access::DeviceKernelRNG::Read<typename T::KernelSpecialisation>;
};


// T = AtomicBlockRNG<REAL>


// Cast to base
// KernelParameter<Access::Read<DeviceKernelRNG<AtomicBlockRNG<double>>>>

// With derived type
// KernelParameter<Access::Read<HostAtomicBlockKernelRNG<double>>>




///**
// *  KernelParameter type for read-only access to a DeviceKernelRNG.
// */
//template <typename T> struct KernelParameter<Access::Read<DeviceKernelRNG<T>>> {
//  using type = Access::DeviceKernelRNG::Read<T>;
//};

///**
// * LoopParameter type for a DeviceKernelRNG.
// */
//template <typename T> struct LoopParameter<Access::Read<DeviceKernelRNG<T>>> {
//  using type = Access::DeviceKernelRNG::Read<T>;
//};


/**
 * The function called before the ParticleLoop is executed.
 */
template <typename T>
inline void pre_loop(ParticleLoopGlobalInfo *global_info,
                     Access::Read<DeviceKernelRNG<T> *> &arg) {
  arg.obj->impl_pre_loop_read(global_info);
}

/**
 * The function called after the ParticleLoop is executed.
 */
template <typename T>
inline void post_loop(ParticleLoopGlobalInfo *global_info,
                      Access::Read<DeviceKernelRNG<T> *> &arg) {
  arg.obj->impl_post_loop_read(global_info);
}

} // namespace ParticleLoopImplementation

/**
 * Abstract base class for RNG implementations which create a block of random
 * numbers on the device prior to the loop execution.
 */
template <typename T> struct DeviceKernelRNG {

   using KernelSpecialisation = T;

  /**
   * Create the loop arguments for the RNG implementation.
   *
   * @param global_info Global information for the loop about to be executed.
   * @returns Pointer to the device data that contains the RNG data.
   */
  virtual inline Access::DeviceKernelRNG::Read<T> impl_get_const(
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
