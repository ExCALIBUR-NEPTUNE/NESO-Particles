#ifndef _NESO_PARTICLES_LOCAL_MEMORY_H_
#define _NESO_PARTICLES_LOCAL_MEMORY_H_

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
class LocalMemory {
public:
  /// Number of bytes required per particle.
  std::size_t size;

  LocalMemory() = default;
  LocalMemory &operator=(const LocalMemory &) = default;

  /**
   * Constructor with specified number of bytes per particle.
   *
   * @param size Number of bytes required per particle.
   */
  LocalMemory(const std::size_t size) : size(size) {
    NESOASSERT(size >= 0, "Invalid local size passed.");
  }

  /**
   * Set the local memory required to a new number of bytes.
   *
   * @param size Number of required bytes per particle.
   */
  inline void set_size(const std::size_t size) { this->size = size; }
};

namespace Access::LocalMemory {

/**
 * ParticleLoop access type for LocalMemory Write access.
 */
struct Write {
  Write() = default;
  void *ptr;
  inline void *data() { return ptr; }
};

} // namespace Access::LocalMemory

namespace ParticleLoopImplementation {

struct LocalMemoryLoopType {
  std::size_t size;
  sycl::local_accessor<std::byte, 1> la;
};

/**
 *  Loop parameter for write access of a LocalMemory.
 */
template <> struct LoopParameter<Access::Write<LocalMemory>> {
  using type = LocalMemoryLoopType;
};

/**
 *  KernelParameter type for write access to a LocalMemory.
 */
template <> struct KernelParameter<Access::Write<LocalMemory>> {
  using type = Access::LocalMemory::Write;
};

/**
 * Method to compute access to a LocalMemory (write)
 */
inline LocalMemoryLoopType create_loop_arg(ParticleLoopGlobalInfo *global_info,
                                           sycl::handler &cgh,
                                           Access::Write<LocalMemory *> &a) {
  LocalMemoryLoopType local_memory_loop;
  const std::size_t size = a.obj->size;
  local_memory_loop.size = size;
  local_memory_loop.la = sycl::local_accessor<std::byte, 1>(
      sycl::range<1>(size * global_info->local_size), cgh);
  return local_memory_loop;
}

/**
 *  Function to create the kernel argument for LocalMemory write access.
 */
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              LocalMemoryLoopType &rhs,
                              Access::LocalMemory::Write &lhs) {
  lhs.ptr =
      static_cast<void *>(&rhs.la[iterationx.local_sycl_index * rhs.size]);
}

/**
 * Indicate that LocalMemory requires local memory.
 */
inline std::size_t
get_required_local_num_bytes([[maybe_unused]] LocalMemory &arg) {
  return arg.size;
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
