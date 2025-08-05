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
    atomic_fetch_add(&ptr[component], value);
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
 * Pre loop execution function for GlobalArray write.
 */
template <typename T>
inline void pre_loop([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                     Access::Add<GlobalArray<T> *> &arg) {
  arg.obj->impl_pre_loop_add();
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
  friend void ParticleLoopImplementation::pre_loop<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      Access::Add<GlobalArray<T> *> &arg);
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
  std::shared_ptr<BufferDevice<T>> d_buffer{nullptr};
  std::shared_ptr<BufferDevice<T>> d_stage_buffer{nullptr};
  bool device_aware_mpi{false};

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline GlobalArrayImplGetT<T> impl_get() {
    NESOASSERT(this->d_stage_buffer != nullptr,
               "Stage buffer has not been set correctly. Make sure that the "
               "GlobalArray is a shared_ptr.");
    return this->d_stage_buffer->ptr;
  }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline GlobalArrayImplGetConstT<T> impl_get_const() {
    return this->d_buffer->ptr;
  }

  /**
   * Pre kernel execution reduction.
   */
  inline void impl_pre_loop_add() {
    this->d_stage_buffer =
        get_resource<BufferDevice<T>, ResourceStackInterfaceBufferDevice<T>>(
            sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<T>{},
            sycl_target);
    this->d_stage_buffer->realloc_no_copy(this->size);
    this->sycl_target->queue
        .fill(static_cast<T *>(this->d_stage_buffer->ptr), static_cast<T>(0),
              this->size)
        .wait_and_throw();
  }

  /**
   * Post kernel execution reduction.
   */
  inline void impl_post_loop_add() {
    NESOASSERT(this->d_stage_buffer != nullptr,
               "Stage buffer has not been set correctly. Make sure that the "
               "GlobalArray is a shared_ptr.");
    auto d_acc_buffer =
        get_resource<BufferDevice<T>, ResourceStackInterfaceBufferDevice<T>>(
            sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<T>{},
            sycl_target);
    d_acc_buffer->realloc_no_copy(this->size);

    T *d_src = this->d_stage_buffer->ptr;
    T *d_acc = d_acc_buffer->ptr;
    T *d_dst = this->d_buffer->ptr;

    this->sycl_target->queue
        .fill(static_cast<T *>(d_acc_buffer->ptr), static_cast<T>(0),
              this->size)
        .wait_and_throw();

    if (this->device_aware_mpi) {
      MPICHK(MPI_Allreduce(d_src, d_acc, static_cast<int>(size),
                           map_ctype_mpi_type<T>(), MPI_SUM, this->comm));
    } else {
      // kernel just completed therefore the data we want is on device
      std::vector<T> tmp_src(this->size);
      std::fill(tmp_src.begin(), tmp_src.end(), (T)0);
      T *t_ptr = tmp_src.data();
      std::vector<T> tmp_acc(this->size);
      std::fill(tmp_acc.begin(), tmp_acc.end(), (T)0);
      T *t_acc_ptr = tmp_acc.data();

      const std::size_t size_bytes = sizeof(T) * this->size;
      sycl_target->queue.memcpy(t_ptr, d_src, size_bytes).wait_and_throw();
      MPICHK(MPI_Allreduce(t_ptr, t_acc_ptr, static_cast<int>(size),
                           map_ctype_mpi_type<T>(), MPI_SUM, this->comm));

      this->sycl_target->queue.memcpy(d_acc, t_acc_ptr, size_bytes)
          .wait_and_throw();
    }

    this->sycl_target->queue
        .parallel_for(sycl::range<1>(this->size),
                      [=](auto ix) { d_dst[ix] += d_acc[ix]; })
        .wait_and_throw();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<T>{}, d_acc_buffer);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<T>{}, this->d_stage_buffer);
    this->d_stage_buffer = nullptr;
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
      : device_aware_mpi(device_aware_mpi_enabled()), sycl_target(sycl_target),
        comm(sycl_target->comm_pair.comm_parent), size(size) {

    this->d_buffer = std::make_shared<BufferDevice<T>>(sycl_target, size);
    if (init_value) {
      this->fill(init_value.value());
    } else {
      this->fill(T());
    }
  }

  /**
   *  Fill the array with a value.
   *
   *  @param value Value to fill the array with - not reduced across MPI ranks.
   */
  inline void fill(const T value) {
    if (this->size > 0) {
      T *ptr = this->d_buffer->ptr;
      sycl_target->queue.fill<T>(ptr, (T)value, this->size).wait_and_throw();
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
      const T *ptr = this->d_buffer->ptr;
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

extern template class GlobalArray<REAL>;
extern template class GlobalArray<INT>;
extern template class GlobalArray<int>;

}; // namespace NESO::Particles

#endif
