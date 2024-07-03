#ifndef _NESO_PARTICLES_KERNEL_RNG_H_
#define _NESO_PARTICLES_KERNEL_RNG_H_

#include "../../loop/particle_loop_base.hpp"
#include "../../loop/particle_loop_index.hpp"
#include "../../particle_group.hpp"

#include <functional>
#include <tuple>

namespace NESO::Particles {

template <typename T> struct KernelRNG;

namespace Access::KernelRNG {

/**
 * This is the kernel type for KernelRNG which is used for all implementations
 * which present RNG values to the kernel via an allocated device buffer.
 */
template <typename T> struct Read {

  /**
   * Stride for matrix access. We assume the matrix is store column wise and a
   * particle accesses a row. This stride is essentially the number of rows
   * (number of particles).
   */
  int stride;

  /**
   * Pointer to the device buffer for all the RNG values.
   */
  T const *RESTRICT d_ptr;

  /**
   * Access the RNG data directly (advanced).
   *
   * @param row Row index to access.
   * @param col Column index to access.
   * @returns Constant reference to RNG data.
   */
  inline const T &at(const int row, const int col) const {
    return d_ptr[col * stride + row];
  }

  /**
   * Access the RNG data for this particle.
   *
   * @param particle_index Particle index to access.
   * @param component RNG component to access.
   * @returns Constant reference to RNG data.
   */
  inline const T &at(const Access::LoopIndex::Read &particle_index,
                     const int component) const {
    const auto index = particle_index.get_loop_linear_index();
    return at(index, component);
  }
};

} // namespace Access::KernelRNG

namespace ParticleLoopImplementation {

/**
 *  KernelParameter type for read-only access to a KernelRNG.
 */
template <typename T> struct KernelParameter<Access::Read<KernelRNG<T>>> {
  using type = Access::KernelRNG::Read<T>;
};

/**
 * LoopParameter type for a KernelRNG.
 */
template <typename T> struct LoopParameter<Access::Read<KernelRNG<T>>> {
  using type = Access::KernelRNG::Read<T>;
};

/**
 * Create the loop argument for a KernelRNG.
 */
template <typename T>
inline Access::KernelRNG::Read<T>
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                sycl::handler &cgh, Access::Read<KernelRNG<T> *> &a) {

  const auto host_arg = a.obj->impl_get_const(global_info);
  Access::KernelRNG::Read<T> kernel_arg = {std::get<0>(host_arg),
                                           std::get<1>(host_arg)};
  return kernel_arg;
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
 * The function called before the ParticleLoop is executed.
 */
template <typename T>
inline void pre_loop(ParticleLoopGlobalInfo *global_info,
                     Access::Read<KernelRNG<T> *> &arg) {
  arg.obj->impl_pre_loop_read(global_info);
}

/**
 * The function called after the ParticleLoop is executed.
 */
template <typename T>
inline void post_loop(ParticleLoopGlobalInfo *global_info,
                      Access::Read<KernelRNG<T> *> &arg) {
  arg.obj->impl_post_loop_read(global_info);
}

} // namespace ParticleLoopImplementation

/**
 * Abstract base class for RNG implementations which create a block of random
 * numbers on the device prior to the loop execution.
 */
template <typename T> struct KernelRNG {

  /**
   * Create the loop arguments for the RNG implementation.
   *
   * @param global_info Global information for the loop about to be executed.
   * @returns Pointer to the device data that contains the RNG data.
   */
  virtual inline std::tuple<int, T *> impl_get_const(
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
};

} // namespace NESO::Particles

#endif
