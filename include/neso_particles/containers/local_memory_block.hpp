#ifndef _NESO_PARTICLES_LOCAL_MEMORY_BLOCK_H_
#define _NESO_PARTICLES_LOCAL_MEMORY_BLOCK_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace NESO::Particles {

/**
 * Type to provide kernel local memory on a per particle basis.
 */
template <typename T> class LocalMemoryBlock {
public:
  /// Number of elements of type T required per particle.
  std::size_t size;

  LocalMemoryBlock() = default;

  /**
   * Constructor with specified number of elements per particle.
   *
   * @param size Number of elements required per particle.
   */
  LocalMemoryBlock(const std::size_t size) : size(size) {}

  /**
   * Set the local memory required to a new number of elements.
   *
   * @param size Number of required elements per particle.
   */
  inline void set_size(const std::size_t size) { this->size = size; }
};

namespace Access::LocalMemoryBlock {

/**
 * ParticleLoop access type for LocalMemoryBlock Write access.
 */
template <typename T> struct Write {
  Write() = default;
  T *ptr;
  inline T *data() { return ptr; }
};

} // namespace Access::LocalMemoryBlock

namespace ParticleLoopImplementation {

template <typename T> struct LocalMemoryLoopType {
  std::size_t size;
  sycl::local_accessor<T, 1> la;
};

/**
 *  Loop parameter for write access of a LocalMemoryBlock.
 */
template <typename T> struct LoopParameter<Access::Write<LocalMemoryBlock<T>>> {
  using type = LocalMemoryLoopType<T>;
};

/**
 *  KernelParameter type for write access to a LocalMemoryBlock.
 */
template <typename T>
struct KernelParameter<Access::Write<LocalMemoryBlock<T>>> {
  using type = Access::LocalMemoryBlock::Write<T>;
};

/**
 * Method to compute access to a LocalMemoryBlock (write)
 */
template <typename T>
inline LocalMemoryLoopType<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<LocalMemoryBlock<T> *> &a) {
  LocalMemoryLoopType<T> local_memory_loop;
  const std::size_t size = a.obj->size;
  local_memory_loop.size = size;
  local_memory_loop.la = sycl::local_accessor<T, 1>(
      sycl::range<1>(size * global_info->local_size), cgh);
  return local_memory_loop;
}

/**
 *  Function to create the kernel argument for LocalMemoryBlock write access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              LocalMemoryLoopType<T> &rhs,
                              Access::LocalMemoryBlock::Write<T> &lhs) {
  lhs.ptr = &rhs.la[iterationx.local_sycl_index * rhs.size];
}

/**
 * Indicate that LocalMemoryBlock requires local memory.
 */
template <typename T>
inline std::size_t
get_required_local_num_bytes(Access::Write<LocalMemoryBlock<T> *> &arg) {
  return arg.obj->size * sizeof(T);
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
