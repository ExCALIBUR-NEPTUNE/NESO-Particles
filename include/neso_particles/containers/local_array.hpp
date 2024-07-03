#ifndef _NESO_PARTICLES_LOCAL_ARRAY_H_
#define _NESO_PARTICLES_LOCAL_ARRAY_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace NESO::Particles {
// Forward declaration of ParticleLoop such that LocalArray can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;
template <typename T> class LocalArray;

template <typename T> using LocalArrayImplGetT = T *;
template <typename T> using LocalArrayImplGetConstT = T const *;

/**
 *  Defines the access implementations and types for LocalArray objects.
 */
namespace Access::LocalArray {

/**
 * Access:LocalArray::Read<T>, Access::LocalArray::Write<T> and
 * Access:LocalArray::Add<T> are the kernel argument types for accessing
 * LocalArray data in a kernel.
 */
/**
 * ParticleLoop access type for LocalArray Read access.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for the array.
  Read() = default;
  T const *ptr;
  const T at(const int component) const { return ptr[component]; }
  const T &operator[](const int component) const { return ptr[component]; }
};

/**
 * ParticleLoop access type for LocalArray Add access.
 */
template <typename T> struct Add {
  /// Pointer to underlying data for the array.
  Add() = default;
  T *ptr;
  /**
   * The local array is local to the MPI rank where the partial sum is a
   * meaningful value.
   */
  inline T fetch_add(const int component, const T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[component]);
    return element_atomic.fetch_add(value);
  }
};

/**
 * ParticleLoop access type for LocalArray Write access.
 */
template <typename T> struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  T *ptr;
  T &at(const int component) { return ptr[component]; }
  T &operator[](const int component) { return ptr[component]; }
};

} // namespace Access::LocalArray

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a LocalArray.
 */
template <typename T> struct LoopParameter<Access::Read<LocalArray<T>>> {
  using type = T const *;
};
/**
 *  Loop parameter for write access of a LocalArray.
 */
template <typename T> struct LoopParameter<Access::Write<LocalArray<T>>> {
  using type = T *;
};
/**
 *  Loop parameter for add access of a LocalArray.
 */
template <typename T> struct LoopParameter<Access::Add<LocalArray<T>>> {
  using type = T *;
};

/**
 *  KernelParameter type for read access to a LocalArray.
 */
template <typename T> struct KernelParameter<Access::Read<LocalArray<T>>> {
  using type = Access::LocalArray::Read<T>;
};
/**
 *  KernelParameter type for write access to a LocalArray.
 */
template <typename T> struct KernelParameter<Access::Write<LocalArray<T>>> {
  using type = Access::LocalArray::Write<T>;
};
/**
 *  KernelParameter type for add access to a LocalArray.
 */
template <typename T> struct KernelParameter<Access::Add<LocalArray<T>>> {
  using type = Access::LocalArray::Add<T>;
};
/**
 *  Function to create the kernel argument for LocalArray read access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx, T const *rhs,
                              Access::LocalArray::Read<T> &lhs) {
  lhs.ptr = rhs;
}
/**
 *  Function to create the kernel argument for LocalArray write access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx, T *rhs,
                              Access::LocalArray::Write<T> &lhs) {
  lhs.ptr = rhs;
}
/**
 *  Function to create the kernel argument for LocalArray add access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx, T *rhs,
                              Access::LocalArray::Add<T> &lhs) {
  lhs.ptr = rhs;
}

/**
 * Method to compute access to a LocalArray (read)
 */
template <typename T>
inline LocalArrayImplGetConstT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<LocalArray<T> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a LocalArray (write)
 */
template <typename T>
inline LocalArrayImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<LocalArray<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a LocalArray (add)
 */
template <typename T>
inline LocalArrayImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Add<LocalArray<T> *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

/**
 * Container to hold an array of values on each MPI rank.
 */
template <typename T> class LocalArray {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;
  friend LocalArrayImplGetConstT<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<LocalArray<T> *> &a);
  friend LocalArrayImplGetT<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<LocalArray<T> *> &a);
  friend LocalArrayImplGetT<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<LocalArray<T> *> &a);

protected:
  std::shared_ptr<BufferDevice<T>> buffer;

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline LocalArrayImplGetT<T> impl_get() { return this->buffer->ptr; }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline LocalArrayImplGetConstT<T> impl_get_const() {
    return this->buffer->ptr;
  }

public:
  /// The SYCLTarget the LocalArray is created on.
  SYCLTargetSharedPtr sycl_target;
  /// The number of elements in the array.
  std::size_t size;

  LocalArray() = default;

  /**
   * Note that the copy operator creates shallow copies of the array.
   */
  LocalArray<T> &operator=(const LocalArray<T> &) = default;

  /**
   *  Create a new LocalArray on a compute target and given size.
   *
   *  @param sycl_target Device to create LocalArray on.
   *  @param size Number of elements in array.
   *  @param Default value to initialise values to.
   */
  LocalArray(SYCLTargetSharedPtr sycl_target, const std::size_t size,
             const std::optional<T> init_value = std::nullopt)
      : sycl_target(sycl_target), size(size) {
    this->buffer = std::make_shared<BufferDevice<T>>(sycl_target, size);
    if (init_value) {
      this->fill(init_value.value());
    }
  }

  /**
   *  Fill the array with a value.
   *
   *  @param value Value to fill the array with.
   */
  inline void fill(const T value) {
    T *ptr = this->buffer->ptr;
    sycl_target->queue.fill(ptr, value, size).wait_and_throw();
  }

  /**
   *  Create a new LocalArray on a compute target and given size.
   *
   *  @param sycl_target Device to create LocalArray on.
   *  @param data Vector to initialise array values to.
   */
  LocalArray(SYCLTargetSharedPtr sycl_target, const std::vector<T> &data)
      : LocalArray(sycl_target, data.size()) {
    this->set(data);
  }

  /**
   * Asynchronously set the values in the local array to those in a std::vector.
   *
   * @param data Input vector to copy values from.
   * @returns Event to wait on before using new values in LocalArray.
   */
  inline sycl::event set_async(const std::vector<T> &data) {
    NESOASSERT(data.size() == this->size, "Input data is incorrectly sized.");
    const std::size_t size_bytes = sizeof(T) * this->size;
    if (size_bytes) {
      auto copy_event = this->sycl_target->queue.memcpy(
          this->buffer->ptr, data.data(), size_bytes);
      return copy_event;
    } else {
      return sycl::event();
    }
  }

  /**
   * Set the values in the local array to those in a std::vector. Blocks until
   * the copy is complete.
   *
   * @param data Input vector to copy values from.
   */
  inline void set(const std::vector<T> &data) {
    this->set_async(data).wait_and_throw();
  }

  /**
   * Asynchronously get the values in the local array into a std::vector.
   *
   * @param[in, out] data Input vector to copy values from LocalArray into.
   * @returns Event to wait on before using new values in the std::vector.
   */
  inline sycl::event get_async(std::vector<T> &data) {
    NESOASSERT(data.size() == this->size, "Input data is incorrectly sized.");
    const std::size_t size_bytes = sizeof(T) * this->size;
    if (size_bytes) {
      auto copy_event = this->sycl_target->queue.memcpy(
          data.data(), this->buffer->ptr, size_bytes);
      return copy_event;
    } else {
      return sycl::event();
    }
  }

  /**
   * Get the values in the local array into a std::vector. Blocks until copy is
   * complete.
   *
   * @param[in, out] data Input vector to copy values from LocalArray into.
   */
  inline void get(std::vector<T> &data) {
    this->get_async(data).wait_and_throw();
  }

  /**
   * Get the values in the local array into a std::vector.
   *
   * @returns std::vector of values in the LocalArray.
   */
  inline std::vector<T> get() {
    std::vector<T> data(this->size);
    this->get(data);
    return data;
  }

  /**
   * Reallocate the buffer to hold the requested number of elements. Current
   * contents is not copied to the new buffer.
   *
   * @param size Number of elements this buffer should  hold.
   */
  inline void realloc_no_copy(const size_t size) {
    this->size = size;
    this->buffer->realloc_no_copy(size);
  }
};

} // namespace NESO::Particles

#endif
