#ifndef _NESO_PARTICLES_LOCAL_MEMORY_INTERLACED_H_
#define _NESO_PARTICLES_LOCAL_MEMORY_INTERLACED_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"

namespace NESO::Particles {

/**
 * Type to provide kernel local memory on a per particle basis which is
 * interlaced between particles in the workgroup.
 */
template <typename T> class LocalMemoryInterlaced {
public:
  /// Number of elements of type T required per particle.
  std::size_t size;

  LocalMemoryInterlaced() = default;

  /**
   * Constructor with specified number of elements per particle. With this
   * container values are interlaced.
   *
   * @param size Number of elements required per particle.
   */
  LocalMemoryInterlaced(const std::size_t size) : size(size) {}

  /**
   * Set the local memory required to a new number of elements.
   *
   * @param size Number of required elements per particle.
   */
  inline void set_size(const std::size_t size) { this->size = size; }
};

namespace Access::LocalMemoryInterlaced {

/**
 * ParticleLoop access type for LocalMemoryInterlaced Write access.
 */
template <typename T> struct Write {
  Write() = default;
  /// Pointer to the start of the local memory for this particle.
  T *ptr;
  /// Stride between elements that this particle can access.
  std::size_t stride;

  /**
   * @returns Pointer to first element in local memory for this particle.
   */
  inline T *data() { return ptr; }

  /**
   * @returns Stride between elements.
   */
  const std::size_t &get_stride() const { return this->stride; };

  /**
   * Access element by component for particle.
   *
   * @param index Index of element to access for particle.
   * @returns Mutable reference to element.
   */
  inline T &at(const int index) { return this->ptr[this->stride * index]; }
};

} // namespace Access::LocalMemoryInterlaced

namespace ParticleLoopImplementation {

template <typename T> struct LocalMemoryInterlacedLoopType {
  std::size_t size;
  sycl::local_accessor<T, 1> la;
};

/**
 *  Loop parameter for write access of a LocalMemoryInterlaced.
 */
template <typename T>
struct LoopParameter<Access::Write<LocalMemoryInterlaced<T>>> {
  using type = LocalMemoryInterlacedLoopType<T>;
};

/**
 *  KernelParameter type for write access to a LocalMemoryInterlaced.
 */
template <typename T>
struct KernelParameter<Access::Write<LocalMemoryInterlaced<T>>> {
  using type = Access::LocalMemoryInterlaced::Write<T>;
};

/**
 * Method to compute access to a LocalMemoryInterlaced (write)
 */
template <typename T>
inline LocalMemoryInterlacedLoopType<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<LocalMemoryInterlaced<T> *> &a) {
  LocalMemoryInterlacedLoopType<T> local_memory_loop;
  const std::size_t size = a.obj->size;
  local_memory_loop.size = size;
  local_memory_loop.la = sycl::local_accessor<T, 1>(
      sycl::range<1>(size * global_info->local_size), cgh);
  return local_memory_loop;
}

/**
 *  Function to create the kernel argument for LocalMemoryInterlaced write
 * access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              LocalMemoryInterlacedLoopType<T> &rhs,
                              Access::LocalMemoryInterlaced::Write<T> &lhs) {
  lhs.ptr = &rhs.la[iterationx.local_sycl_index];
  lhs.stride = iterationx.local_sycl_range;
}

/**
 * Indicate that LocalMemoryInterlaced requires local memory.
 */
template <typename T>
inline std::size_t
get_required_local_num_bytes(Access::Write<LocalMemoryInterlaced<T> *> &arg) {
  return arg.obj->size * sizeof(T);
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
