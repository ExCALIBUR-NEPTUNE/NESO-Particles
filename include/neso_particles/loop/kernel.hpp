#ifndef __NESO_PARTICLES_LOOP_KERNEL_HPP_
#define __NESO_PARTICLES_LOOP_KERNEL_HPP_

#include "../compute_target.hpp"

namespace NESO::Particles {

namespace Kernel {

/**
 * Type for holding the number of bytes read and written by a kernel.
 */
struct NumBytes {
  std::size_t value{0};
  NumBytes() = default;
  NumBytes(const std::size_t value) : value(value) {}
};

/**
 * Type for holding the number of FLOPs a kernel performs.
 */
struct NumFLOP {
  std::size_t value{0};
  NumFLOP() = default;
  NumFLOP(const std::size_t value) : value(value) {}
};

/**
 * Container to store the metadata for a kernel.
 */
class Metadata {
protected:
  inline void unpack_arg(NumBytes &arg) { this->num_bytes = arg; }
  inline void unpack_arg(NumFLOP &arg) { this->num_flops = arg; }
  template <typename T> inline void recurse_args(T first) {
    this->unpack_arg(first);
  }
  template <typename T, typename... ARGS>
  inline void recurse_args(T first, ARGS... args) {
    this->unpack_arg(first);
    this->recurse_args(args...);
  }

public:
  /// The number of bytes moved by a single execution of the kernel.
  NumBytes num_bytes;
  /// The number of FLOP performed by a single execution of the kernel.
  NumFLOP num_flops;

  Metadata() = default;

  /**
   * Create a metadata store from a collection of attributes.
   *
   * @param args NumBytes and NumFLOP instances.
   */
  template <typename... ARGS> Metadata(ARGS... args) {
    this->recurse_args(args...);
  }
};

/**
 * This is a container to wrap a device copyable callable type as the kernel
 * function along with user provided metadata.
 */
template <typename KERNEL_TYPE> struct Kernel {
  /// The kernel for the parallel loop.
  KERNEL_TYPE kernel;
  /// Metadata that accompanies the kernel.
  Metadata metadata;
  /**
   *  Wrap a kernel without any metadata.
   *
   *  @param kernel Device copyable callable to use as a kernel.
   */
  Kernel(KERNEL_TYPE kernel) : kernel(kernel) {}
  /**
   *  Wrap a kernel with metadata.
   *
   *  @param kernel Device copyable callable to use as a kernel.
   *  @param metadata Metadata for the kernel.
   */
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
 * Extract the number of bytes per kernel invocation from the kernel
 * metadata.
 *
 * @param kernel Device copyable callable type to use as kernel.
 * @returns Number of bytes in kernel metadata.
 */
template <typename T>
inline std::size_t get_kernel_num_bytes(Kernel::Kernel<T> &kernel) {
  return kernel.metadata.num_bytes.value;
}

/**
 * Extract the number of flops per kernel invocation from the kernel
 * metadata.
 *
 * @param kernel Device copyable callable type to use as kernel.
 * @returns Number flops in kernel metadata.
 */
template <typename T>
inline std::size_t get_kernel_num_flops(Kernel::Kernel<T> &kernel) {
  return kernel.metadata.num_flops.value;
}

}; // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
