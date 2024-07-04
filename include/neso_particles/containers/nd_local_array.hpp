#ifndef _NESO_PARTICLES_ND_LOCAL_ARRAY_H_
#define _NESO_PARTICLES_ND_LOCAL_ARRAY_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "nd_index.hpp"
#include "tuple.hpp"

namespace NESO::Particles {

// Forward declaration of ParticleLoop such that NDLocalArray can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;
template <typename T, std::size_t N> class NDLocalArray;

/**
 *  Defines the access implementations and types for NDLocalArray objects.
 */
namespace Access::NDLocalArray {

/**
 * ParticleLoop access type for NDLocalArray Read access.
 */
template <typename T, std::size_t N> struct Read {
  // Pointer to underlying data for the array.
  Read() = default;
  T const *RESTRICT ptr;
  NDIndex<N> index;
  template <typename... I> const T &at(I... ix) const {
    return ptr[index.get_linear_index(ix...)];
  }
};

/**
 * ParticleLoop access type for NDLocalArray Write access.
 */
template <typename T, std::size_t N> struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  T *RESTRICT ptr;
  NDIndex<N> index;
  template <typename... I> inline T &at(I... ix) {
    return ptr[index.get_linear_index(ix...)];
  }
};

/**
 * ParticleLoop access type for NDLocalArray Add access.
 */
template <typename T, std::size_t N> struct Add {
  /// Pointer to underlying data for the array.
  Add() = default;
  T *RESTRICT ptr;
  NDIndex<N> index;
  template <typename... I> inline T fetch_add(I... ix) {
    auto tuple_index = Tuple::to_tuple(ix...);

    auto lambda_index_wrapper = [&](auto... ax) {
      return this->index.get_linear_index(ax...);
    };
    const auto index =
        Tuple::apply_truncated<N>(lambda_index_wrapper, tuple_index);

    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[index]);

    const auto value = Tuple::get_last_arg(ix...);
    return element_atomic.fetch_add(value);
  }
};

} // namespace Access::NDLocalArray

namespace ParticleLoopImplementation {

/**
 *  KernelParameter type for read access to a NDLocalArray.
 */
template <typename T, std::size_t N>
struct KernelParameter<Access::Read<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Read<T, N>;
};
/**
 *  KernelParameter type for write access to a NDLocalArray.
 */
template <typename T, std::size_t N>
struct KernelParameter<Access::Write<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Write<T, N>;
};
/**
 *  KernelParameter type for add access to a NDLocalArray.
 */
template <typename T, std::size_t N>
struct KernelParameter<Access::Add<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Add<T, N>;
};

/**
 *  Loop parameter for read access of a NDLocalArray.
 */
template <typename T, std::size_t N>
struct LoopParameter<Access::Read<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Read<T, N>;
};
/**
 *  Loop parameter for write access of a NDLocalArray.
 */
template <typename T, std::size_t N>
struct LoopParameter<Access::Write<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Write<T, N>;
};
/**
 *  Loop parameter for add access of a NDLocalArray.
 */
template <typename T, std::size_t N>
struct LoopParameter<Access::Add<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Add<T, N>;
};

/**
 * Method to compute access to a NDLocalArray (read)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Read<T, N>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get_const(), a.obj->index};
}

/**
 * Method to compute access to a NDLocalArray (write)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Write<T, N>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get(), a.obj->index};
}

/**
 * Method to compute access to a NDLocalArray (add)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Add<T, N>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Add<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get(), a.obj->index};
}

/**
 *  Function to create the kernel argument for NDLocalArray read access.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              Access::NDLocalArray::Read<T, N> &rhs,
                              Access::NDLocalArray::Read<T, N> &lhs) {
  lhs = rhs;
}
/**
 *  Function to create the kernel argument for NDLocalArray write access.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              Access::NDLocalArray::Write<T, N> &rhs,
                              Access::NDLocalArray::Write<T, N> &lhs) {
  lhs = rhs;
}
/**
 *  Function to create the kernel argument for NDLocalArray add access.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              Access::NDLocalArray::Add<T, N> &rhs,
                              Access::NDLocalArray::Add<T, N> &lhs) {
  lhs = rhs;
}

} // namespace ParticleLoopImplementation

/**
 * Generic N-Dimensional array type which is accessible on the host and in a
 * @ref ParticleLoop kernel.
 */
template <typename T, std::size_t N> class NDLocalArray {

  friend Access::NDLocalArray::Read<T, N>
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<NDLocalArray<T, N> *> &a);
  friend Access::NDLocalArray::Write<T, N>
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<NDLocalArray<T, N> *> &a);
  friend Access::NDLocalArray::Add<T, N>
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<NDLocalArray<T, N> *> &a);

protected:
  std::shared_ptr<BufferDevice<T>> buffer;
  INT size;

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T *impl_get() { return this->buffer->ptr; }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T *impl_get_const() { return this->buffer->ptr; }

public:
  NDLocalArray() = default;

  /// Compute device for the array.
  SYCLTargetSharedPtr sycl_target;

  /// Indexing function for the NDLocalArray.
  NDIndex<N> index;

  /**
   * Create a NDLocalArray on a compute device with a given shape.
   *
   * @param sycl_target Compute device to create local array on.
   * @param shape Parameter pack of size N which defines the extent of the
   * array in each of the N dimensions.
   */
  template <typename... SHAPE>
  NDLocalArray(SYCLTargetSharedPtr sycl_target, SHAPE... shape)
      : sycl_target(sycl_target) {
    this->index = nd_index<N>(shape...);
    this->size = this->index.size();
    this->buffer =
        std::make_shared<BufferDevice<T>>(this->sycl_target, this->size);
  }

  /**
   *  Fill the array with a value.
   *
   *  @param value Value to fill the array with.
   */
  inline void fill(const T value) {
    T *ptr = this->buffer->ptr;
    sycl_target->queue.fill(ptr, value, this->size).wait_and_throw();
  }

  /**
   * Asynchronously set the values in the local array to those in a std::vector.
   *
   * @param data Input vector to copy values from.
   * @returns Event to wait on before using new values in NDLocalArray.
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
   * @param[in, out] data Input vector to copy values from NDLocalArray into.
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
   * @param[in, out] data Input vector to copy values from NDLocalArray into.
   */
  inline void get(std::vector<T> &data) {
    this->get_async(data).wait_and_throw();
  }

  /**
   * Get the values in the local array into a std::vector.
   *
   * @returns std::vector of values in the NDLocalArray.
   */
  inline std::vector<T> get() {
    std::vector<T> data(this->size);
    this->get(data);
    return data;
  }
};

} // namespace NESO::Particles

#endif
