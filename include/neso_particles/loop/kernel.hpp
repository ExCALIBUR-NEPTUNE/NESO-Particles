#ifndef __NESO_PARTICLES_LOOP_KERNEL_HPP_
#define __NESO_PARTICLES_LOOP_KERNEL_HPP_

#include "../compute_target.hpp"

namespace NESO::Particles {

// TODO API DOCS FOR THIS FILE
namespace Kernel {

struct LocalSize {
  std::size_t value{0};
  LocalSize() = default;
  LocalSize(const std::size_t value) : value(value) {}
};

struct NumBytes {
  std::size_t value{0};
  NumBytes() = default;
  NumBytes(const std::size_t value) : value(value) {}
};

struct NumFlops {
  std::size_t value{0};
  NumFlops() = default;
  NumFlops(const std::size_t value) : value(value) {}
};

class Metadata {
protected:
  inline void unpack_arg(LocalSize &arg) { this->local_size = arg; }
  inline void unpack_arg(NumBytes &arg) { this->num_bytes = arg; }
  inline void unpack_arg(NumFlops &arg) { this->num_flops = arg; }
  template <typename T> inline void recurse_args(T first) {
    this->unpack_arg(first);
  }
  template <typename T, typename... ARGS>
  inline void recurse_args(T first, ARGS... args) {
    this->unpack_arg(first);
    this->recurse_args(args...);
  }

public:
  LocalSize local_size;
  NumBytes num_bytes;
  NumFlops num_flops;

  Metadata() = default;

  template <typename... ARGS> Metadata(ARGS... args) {
    this->recurse_args(args...);
  }
};

template <typename KERNEL_TYPE> struct Kernel {
  KERNEL_TYPE kernel;
  Metadata metadata;
  Kernel(KERNEL_TYPE kernel) : kernel(kernel) {}
  Kernel(KERNEL_TYPE kernel, Metadata metadata)
      : kernel(kernel), metadata(metadata) {}
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
  return kernel.metadata.num_bytes.value;
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
  return kernel.metadata.num_flops.value;
}

}; // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
