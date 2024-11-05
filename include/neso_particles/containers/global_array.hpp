#ifndef _NESO_PARTICLES_GLOBAL_ARRAY_H_
#define _NESO_PARTICLES_GLOBAL_ARRAY_H_

#include "../compute_target.hpp"
#include "../typedefs.hpp"

#include "../communication.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include <memory>
#include <mpi.h>
#include <optional>
#include <vector>

namespace NESO::Particles {

// Forward declaration of ParticleLoop such that GlobalArray can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;
template <typename T> class GlobalArray;
class ParticleGroup;
template <typename T> using GlobalArrayImplGetT = T *;
template <typename T> using GlobalArrayImplGetConstT = T const *;

/**
 *  Defines the access implementations and types for GlobalArray objects.
 */
namespace Access::GlobalArray {

/**
 * Access:GlobalArray::Read<T> and Access:GlobalArray::Add<T> are the
 * kernel argument types for accessing GlobalArray data in a kernel.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for the array.
  Read() = default;
  T const *ptr;
  inline const T at(const int component) { return ptr[component]; }
  inline const T &operator[](const int component) { return ptr[component]; }
};

/**
 * Access:GlobalArray::Read<T> and Access:GlobalArray::Add<T> are the
 * kernel argument types for accessing GlobalArray data in a kernel.
 */
template <typename T> struct Add {
  /// Pointer to underlying data for the array.
  Add() = default;
  T *ptr;
  /**
   * This does not return a value as the returned value would be a partial sum
   * on this MPI rank.
   */
  inline void add(const int component, const T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[component]);
    element_atomic.fetch_add(value);
  }
};

} // namespace Access::GlobalArray

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a GlobalArray.
 */
template <typename T> struct LoopParameter<Access::Read<GlobalArray<T>>> {
  using type = T const *;
};
/**
 *  Loop parameter for add access of a GlobalArray.
 */
template <typename T> struct LoopParameter<Access::Add<GlobalArray<T>>> {
  using type = T *;
};

/**
 *  KernelParameter type for read access to a GlobalArray.
 */
template <typename T> struct KernelParameter<Access::Read<GlobalArray<T>>> {
  using type = Access::GlobalArray::Read<T>;
};
/**
 *  KernelParameter type for add access to a GlobalArray.
 */
template <typename T> struct KernelParameter<Access::Add<GlobalArray<T>>> {
  using type = Access::GlobalArray::Add<T>;
};

/**
 *  Function to create the kernel argument for GlobalArray read access.
 */
template <typename T>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  T const *rhs, Access::GlobalArray::Read<T> &lhs) {
  lhs.ptr = rhs;
}
/**
 *  Function to create the kernel argument for GlobalArray add access.
 */
template <typename T>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx, T *rhs,
                  Access::GlobalArray::Add<T> &lhs) {
  lhs.ptr = rhs;
}

/**
 * Post loop execution function for GlobalArray write.
 */
template <typename T>
inline void post_loop([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                      Access::Add<GlobalArray<T> *> &arg) {
  arg.obj->impl_post_loop_add();
}

/**
 * Method to compute access to a GlobalArray (read)
 */
template <typename T>
inline GlobalArrayImplGetConstT<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<GlobalArray<T> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a GlobalArray (add)
 */
template <typename T>
inline GlobalArrayImplGetT<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Add<GlobalArray<T> *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

/**
 *  GlobalArray is an array type which can be accessed from kernels in read or
 *  atomic add mode. Post loop execution, with add access mode, the global
 *  array values are automatically reduced across the MPI communicator.
 */
template <typename T> class GlobalArray {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;
  friend void ParticleLoopImplementation::post_loop<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      Access::Add<GlobalArray<T> *> &arg);
  friend GlobalArrayImplGetConstT<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<GlobalArray<T> *> &a);
  friend GlobalArrayImplGetT<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<GlobalArray<T> *> &a);

protected:
  std::shared_ptr<BufferDeviceHost<T>> buffer;

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline GlobalArrayImplGetT<T> impl_get() {
    return this->buffer->d_buffer.ptr;
  }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline GlobalArrayImplGetConstT<T> impl_get_const() {
    return this->buffer->d_buffer.ptr;
  }

  /**
   * Post kernel execution reduction.
   */
  inline void impl_post_loop_add() {

    // kernel just completed therefore the data we want is on device
    std::vector<T> tmp(this->size);
    T *t_ptr = tmp.data();
    T *d_ptr = this->buffer->d_buffer.ptr;
    T *h_ptr = this->buffer->h_buffer.ptr;
    const std::size_t size_bytes = sizeof(T) * this->size;
    sycl_target->queue.memcpy(t_ptr, d_ptr, size_bytes).wait_and_throw();

    MPICHK(MPI_Allreduce(t_ptr, h_ptr, static_cast<int>(size),
                         map_ctype_mpi_type<T>(), MPI_SUM, this->comm));
    this->buffer->host_to_device();
  }

public:
  /// The SYCLTarget the GlobalArray is created on.
  SYCLTargetSharedPtr sycl_target;
  /// The MPI communicator reductions are made over.
  MPI_Comm comm;
  /// The number of elements in the array.
  std::size_t size = {0};

  GlobalArray() = default;

  /**
   * Note that the copy operator creates shallow copies of the array.
   */
  GlobalArray<T> &operator=(const GlobalArray<T> &) = default;
  GlobalArray<T>(const GlobalArray<T> &) = default;

  /**
   *  Create a new GlobalArray on a compute target and given size.
   *
   *  @param sycl_target Device to create GlobalArray on.
   *  @param size Number of elements in array.
   *  @param Default value to initialise values to.
   */
  GlobalArray(SYCLTargetSharedPtr sycl_target, const std::size_t size,
              const std::optional<T> init_value = std::nullopt)
      : sycl_target(sycl_target), comm(sycl_target->comm_pair.comm_parent),
        size(size) {

    this->buffer = std::make_shared<BufferDeviceHost<T>>(sycl_target, size);
    if (init_value) {
      this->fill(init_value.value());
    }
  }

  /**
   *  Fill the array with a value.
   *
   *  @param value Value to fill the array with - not reduced across MPI ranks.
   */
  inline void fill(const T value) {
    if (this->size > 0) {
      T *ptr = this->buffer->d_buffer.ptr;
      auto e0 = sycl_target->queue.fill<T>(ptr, value, this->size);

      auto h_ptr = this->buffer->h_buffer.ptr;
      for (std::size_t ix = 0; ix < this->size; ix++) {
        h_ptr[ix] = value;
      }
      e0.wait_and_throw();
    }
  }

  /**
   * Asynchronously get the values in the local array into a std::vector.
   *
   * @param[in, out] data Input vector to copy values from GlobalArray into.
   * @returns Event to wait on before using new values in the std::vector.
   */
  inline sycl::event get_async(std::vector<T> &data) {
    NESOASSERT(data.size() == this->size, "Input data is incorrectly sized.");
    const std::size_t size_bytes = sizeof(T) * this->size;
    if (size_bytes) {
      const T *ptr = this->buffer->h_buffer.ptr;
      auto copy_event = sycl_target->queue.memcpy(data.data(), ptr, size_bytes);
      return copy_event;
    } else {
      return sycl::event();
    }
  }

  /**
   * Get the values in the local array into a std::vector. Blocks until copy is
   * complete.
   *
   * @param[in, out] data Input vector to copy values from GlobalArray into.
   */
  inline void get(std::vector<T> &data) {
    this->get_async(data).wait_and_throw();
  }

  /**
   * Get the values in the local array into a std::vector.
   *
   * @returns std::vector of values in the GlobalArray.
   */
  inline std::vector<T> get() {
    std::vector<T> data(this->size);
    this->get(data);
    return data;
  }
};

}; // namespace NESO::Particles

#endif
