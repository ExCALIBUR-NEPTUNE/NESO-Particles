#ifndef _NESO_PARTICLES_ND_LOCAL_ARRAY_H_
#define _NESO_PARTICLES_ND_LOCAL_ARRAY_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../nd_host_array.hpp"
#include "../pair_loop/particle_pair_loop_base.hpp"
#include "nd_index.hpp"
#include "rng/rng_generation_function.hpp"
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
  T *RESTRICT ptr;
  NDIndex<N> index;
  template <typename... I> inline T fetch_add(I... ix) {
    auto tuple_index = Tuple::to_tuple(ix...);

    auto lambda_index_wrapper = [&](auto... ax) {
      return this->index.get_linear_index(ax...);
    };
    const auto index =
        Tuple::apply_truncated<N>(lambda_index_wrapper, tuple_index);

    const T value = Tuple::get_last_arg(ix...);
    return atomic_fetch_add(&ptr[index], value);
  }
};

/**
 * ParticleLoop access type for NDLocalArray Max access.
 */
template <typename T, std::size_t N> struct Max {
  /// Pointer to underlying data for the array.
  T *RESTRICT ptr;
  NDIndex<N> index;
  template <typename... I> inline T fetch_max(I... ix) {
    auto tuple_index = Tuple::to_tuple(ix...);

    auto lambda_index_wrapper = [&](auto... ax) {
      return this->index.get_linear_index(ax...);
    };
    const auto index =
        Tuple::apply_truncated<N>(lambda_index_wrapper, tuple_index);

    const T value = Tuple::get_last_arg(ix...);
    return atomic_fetch_max(&ptr[index], value);
  }
};

/**
 * ParticleLoop access type for NDLocalArray Min access.
 */
template <typename T, std::size_t N> struct Min {
  /// Pointer to underlying data for the array.
  T *RESTRICT ptr;
  NDIndex<N> index;
  template <typename... I> inline T fetch_min(I... ix) {
    auto tuple_index = Tuple::to_tuple(ix...);

    auto lambda_index_wrapper = [&](auto... ax) {
      return this->index.get_linear_index(ax...);
    };
    const auto index =
        Tuple::apply_truncated<N>(lambda_index_wrapper, tuple_index);

    const T value = Tuple::get_last_arg(ix...);
    return atomic_fetch_min(&ptr[index], value);
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
 *  KernelParameter type for max access to a NDLocalArray.
 */
template <typename T, std::size_t N>
struct KernelParameter<Access::Max<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Max<T, N>;
};

/**
 *  KernelParameter type for min access to a NDLocalArray.
 */
template <typename T, std::size_t N>
struct KernelParameter<Access::Min<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Min<T, N>;
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
 *  Loop parameter for max access of a NDLocalArray.
 */
template <typename T, std::size_t N>
struct LoopParameter<Access::Max<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Max<T, N>;
};

/**
 *  Loop parameter for min access of a NDLocalArray.
 */
template <typename T, std::size_t N>
struct LoopParameter<Access::Min<NDLocalArray<T, N>>> {
  using type = Access::NDLocalArray::Min<T, N>;
};

/**
 * Method to compute access to a NDLocalArray (read)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Read<T, N>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get_const(), a.obj->index};
}

/**
 * Method to compute access to a NDLocalArray (write)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Write<T, N>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get(), a.obj->index};
}

/**
 * Method to compute access to a NDLocalArray (add)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Add<T, N>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Add<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get(), a.obj->index};
}

/**
 * Method to compute access to a NDLocalArray (max)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Max<T, N>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Max<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get(), a.obj->index};
}

/**
 * Method to compute access to a NDLocalArray (min)
 */
template <typename T, std::size_t N>
inline Access::NDLocalArray::Min<T, N>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Min<NDLocalArray<T, N> *> &a) {
  return {a.obj->impl_get(), a.obj->index};
}

/**
 *  Function to create the kernel argument for NDLocalArray read access.
 */
template <typename T, std::size_t N>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::NDLocalArray::Read<T, N> &rhs,
                  Access::NDLocalArray::Read<T, N> &lhs) {
  lhs = rhs;
}
/**
 *  Function to create the kernel argument for NDLocalArray write access.
 */
template <typename T, std::size_t N>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::NDLocalArray::Write<T, N> &rhs,
                  Access::NDLocalArray::Write<T, N> &lhs) {
  lhs = rhs;
}
/**
 *  Function to create the kernel argument for NDLocalArray add access.
 */
template <typename T, std::size_t N>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::NDLocalArray::Add<T, N> &rhs,
                  Access::NDLocalArray::Add<T, N> &lhs) {
  lhs = rhs;
}

/**
 *  Function to create the kernel argument for NDLocalArray max access.
 */
template <typename T, std::size_t N>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::NDLocalArray::Max<T, N> &rhs,
                  Access::NDLocalArray::Max<T, N> &lhs) {
  lhs = rhs;
}

/**
 *  Function to create the kernel argument for NDLocalArray min access.
 */
template <typename T, std::size_t N>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::NDLocalArray::Min<T, N> &rhs,
                  Access::NDLocalArray::Min<T, N> &lhs) {
  lhs = rhs;
}

} // namespace ParticleLoopImplementation

namespace ParticlePairLoopImplementation {

/**
 *  Function to create the kernel argument for NDLocalArray read
 * access in a pair loop.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    Access::NDLocalArray::Read<T, N> &rhs,
    Access::NDLocalArray::Read<T, N> &lhs) {

  lhs = rhs;
}

/**
 *  Function to create the kernel argument for NDLocalArray write
 * access in a pair loop.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    Access::NDLocalArray::Write<T, N> &rhs,
    Access::NDLocalArray::Write<T, N> &lhs) {

  lhs = rhs;
}

/**
 *  Function to create the kernel argument for NDLocalArray add
 * access in a pair loop.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    Access::NDLocalArray::Add<T, N> &rhs,
    Access::NDLocalArray::Add<T, N> &lhs) {

  lhs = rhs;
}

/**
 *  Function to create the kernel argument for NDLocalArray max
 * access in a pair loop.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    Access::NDLocalArray::Max<T, N> &rhs,
    Access::NDLocalArray::Max<T, N> &lhs) {

  lhs = rhs;
}
/**
 *  Function to create the kernel argument for NDLocalArray min
 * access in a pair loop.
 */
template <typename T, std::size_t N>
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    Access::NDLocalArray::Min<T, N> &rhs,
    Access::NDLocalArray::Min<T, N> &lhs) {

  lhs = rhs;
}

} // namespace ParticlePairLoopImplementation

/**
 * Generic N-Dimensional array type which is accessible on the host and in a
 * @ref ParticleLoop kernel.
 */
template <typename T, std::size_t N> class NDLocalArray {

  friend Access::NDLocalArray::Read<T, N>
  ParticleLoopImplementation::create_loop_arg<T, N>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<NDLocalArray<T, N> *> &a);
  friend Access::NDLocalArray::Write<T, N>
  ParticleLoopImplementation::create_loop_arg<T, N>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<NDLocalArray<T, N> *> &a);
  friend Access::NDLocalArray::Add<T, N>
  ParticleLoopImplementation::create_loop_arg<T, N>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<NDLocalArray<T, N> *> &a);
  friend Access::NDLocalArray::Max<T, N>
  ParticleLoopImplementation::create_loop_arg<T, N>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Max<NDLocalArray<T, N> *> &a);
  friend Access::NDLocalArray::Min<T, N>
  ParticleLoopImplementation::create_loop_arg<T, N>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Min<NDLocalArray<T, N> *> &a);

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
   * @param index Specification of the extent of each dimension.
   */
  NDLocalArray(SYCLTargetSharedPtr sycl_target, NDIndex<N> index)
      : sycl_target(sycl_target), index(index) {
    this->size = this->index.size();
    this->buffer =
        std::make_shared<BufferDevice<T>>(this->sycl_target, this->size);
    this->fill(T());
  }

  /**
   * Create a NDLocalArray on a compute device from a NDHostArray.
   *
   * @param sycl_target Compute device to create local array on.
   * @param nd_host_array NDHostArray to copy to device.
   */
  NDLocalArray(SYCLTargetSharedPtr sycl_target,
               NDHostArraySharedPtr<T, N> &nd_host_array)
      : NDLocalArray(sycl_target, nd_host_array->index) {
    this->set(nd_host_array);
  }

  /**
   * Create a NDLocalArray on a compute device with a given shape.
   *
   * @param sycl_target Compute device to create local array on.
   * @param shape Parameter pack of size N which defines the extent of the
   * array in each of the N dimensions.
   */
  template <typename... SHAPE>
  NDLocalArray(SYCLTargetSharedPtr sycl_target, SHAPE... shape)
      : NDLocalArray(sycl_target, nd_index<N>(shape...)) {
    static_assert(sizeof...(shape) == N, "Missmatch between shape size and N.");
  }

  /**
   *  Fill the array with a value.
   *
   *  @param value Value to fill the array with.
   */
  inline void fill(const T value) {
    T *ptr = this->buffer->ptr;
    if (this->size > 0) {
      sycl_target->queue.fill(ptr, value, this->size).wait_and_throw();
    }
  }

  /**
   * Fill the array from samples generated by a RNG generation function that
   * conforms to the RNGGenerationFunction interface.
   *
   * @param rng_generation_function RNG sampler to use.
   */
  inline void
  fill(std::shared_ptr<RNGGenerationFunction<T>> rng_generation_function) {
    rng_generation_function->draw_random_samples(this->sycl_target, this->ptr(),
                                                 this->size, 8192);
  }

  /**
   * @returns Pointer to underlying data. Data is linearised slowest to fastest.
   */
  inline T *ptr() { return this->impl_get(); }

  /**
   * Asynchronously set the values in the local array to those in a std::vector.
   *
   * @param data Input vector to copy values from.
   * @returns Event to wait on before using new values in NDLocalArray.
   */
  inline sycl::event set_async(const std::vector<T> &data) {
    NESOASSERT(data.size() == static_cast<std::size_t>(this->size),
               "Input data is incorrectly sized.");
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
   * Copy the values from the NDHostArray into an NDLocalArray.
   *
   * @param nd_host_array Source array.
   */
  inline void set(NDHostArraySharedPtr<T, N> &nd_host_array) {

    NESOASSERT(nd_host_array->index == this->index,
               "Shape of passed NDHostArray does not match the shape of the "
               "NDLocalArray");

    this->sycl_target->queue
        .memcpy(this->buffer->ptr, nd_host_array->ptr(), this->size * sizeof(T))
        .wait_and_throw();
  }

  /**
   * Asynchronously get the values in the local array into a std::vector.
   *
   * @param[in, out] data Input vector to copy values from NDLocalArray into.
   * @returns Event to wait on before using new values in the std::vector.
   */
  inline sycl::event get_async(std::vector<T> &data) {
    NESOASSERT(data.size() == static_cast<std::size_t>(this->size),
               "Input data is incorrectly sized.");
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

  /**
   * Copy the values from the NDLocalArray into an NDHostArray.
   *
   * @param[in, out] nd_host_array Destination array, will be allocated if
   * nullptr.
   */
  inline void get(NDHostArraySharedPtr<T, N> &nd_host_array) {
    if (nd_host_array == nullptr) {
      nd_host_array =
          NESO::Particles::nd_host_array<T, N>(this->sycl_target, this->index);
    } else {
      NESOASSERT(nd_host_array->index == this->index,
                 "Shape of passed NDHostArray does not match the shape of the "
                 "NDLocalArray");
    }

    this->sycl_target->queue
        .memcpy(nd_host_array->ptr(), this->buffer->ptr, this->size * sizeof(T))
        .wait_and_throw();
  }

  /**
   * Update each held entry a as
   *
   * a <- binop(a, b)
   *
   * where the entries b are supplied by another NDLocalArray and binop is a
   * provided binary operator.
   *
   * @param second_array Second array, i.e. b, for the binary combination.
   * @param binop Binary operation to use to combine elements, must be a device
   * copyable object with a cell method that takes two arguments of type T and
   * U.
   */
  template <typename U, typename BINOP>
  inline void combine(std::shared_ptr<NDLocalArray<U, N>> second_array,
                      BINOP binop) {

    NESOASSERT(second_array.get() != this,
               "Second array is the same as this array.");
    NESOASSERT(this->index == second_array->index,
               "Passed array has different dimension extents to this array.");

    T *RESTRICT k_a = this->buffer->ptr;
    U const *RESTRICT const k_b = second_array->buffer->ptr;
    BINOP k_binop = binop;

    this->sycl_target->queue
        .parallel_for(this->sycl_target->device_limits.validate_range_global(
                          sycl::range<1>(this->size)),
                      [=](auto idx) { k_a[idx] = k_binop(k_a[idx], k_b[idx]); })
        .wait_and_throw();
  }
};

extern template class NDLocalArray<REAL, 2>;
extern template class NDLocalArray<INT, 2>;
extern template class NDLocalArray<int, 2>;
extern template class NDLocalArray<REAL, 3>;
extern template class NDLocalArray<INT, 3>;
extern template class NDLocalArray<int, 3>;

template <typename T, std::size_t N>
using NDLocalArraySharedPtr = std::shared_ptr<NDLocalArray<T, N>>;

} // namespace NESO::Particles

#endif
