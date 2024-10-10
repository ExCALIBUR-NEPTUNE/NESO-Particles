#ifndef __NESO_PARTICLES_LOOP_KERNEL_HPP_
#define __NESO_PARTICLES_LOOP_KERNEL_HPP_

#include "../compute_target.hpp"

namespace NESO::Particles {

namespace Kernel {

struct MetaData {
  std::size_t local_size{0};
  std::size_t num_bytes{0};
  std::size_t num_flops{0};
};

template <typename KERNEL_TYPE> struct Kernel {
  KERNEL_TYPE kernel;
  MetaData meta_data;
  Kernel(KERNEL_TYPE kernel) : kernel(kernel) {}
  Kernel(KERNEL_TYPE kernel, MetaData meta_data)
      : kernel(kernel), meta_data(meta_data) {}
};

} // namespace Kernel

namespace ParticleLoopImplementation {

/**
 * Extract the kernel from an object for a ParticleLoop.
 * @param kernel Device copyable callable type to use as kernel.
 * @returns kernel.
 */
template <typename T> inline T &get_kernel(Kernel::Kernel<T> &kernel) {
  return kernel.kernel;
}

/**
 * Extract the number of bytes per kernel invocation from the kernel. No-op
 * implementation for when kernel is a generic callable.
 *
 * @param kernel Device copyable callable type to use as kernel.
 * @returns 0.
 */
template <typename T>
inline std::size_t get_kernel_num_bytes(Kernel::Kernel<T> &kernel) {
  return kernel.meta_data.num_bytes;
}

/**
 * Extract the number of flops per kernel invocation from the kernel. No-op
 * implementation for when kernel is a generic callable.
 *
 * @param kernel Device copyable callable type to use as kernel.
 * @returns 0.
 */
template <typename T>
inline std::size_t get_kernel_num_flops(Kernel::Kernel<T> &kernel) {
  return kernel.meta_data.num_flops;
}

}; // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
