#ifndef _NESO_PARTICLES_LOCAL_ARRAY_H_
#define _NESO_PARTICLES_LOCAL_ARRAY_H_

#include "../compute_target.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace NESO::Particles {
// Forward declaration of ParticleLoop such that LocalArray can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;

/**
 * Container to hold an array of values on each MPI rank.
 */
template <typename T> class LocalArray {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;

protected:
  std::shared_ptr<BufferDevice<T>> buffer;

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T *impl_get() { return this->buffer->ptr; }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T const *impl_get_const() { return this->buffer->ptr; }

public:
  /// The SYCLTarget the LocalArray is created on.
  SYCLTargetSharedPtr sycl_target;
  /// The number of elements in the array.
  std::size_t size;

  LocalArray() = default;
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
      auto copy_event =
          sycl_target->queue.memcpy(this->buffer->ptr, data.data(), size_bytes);
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
      auto copy_event =
          sycl_target->queue.memcpy(data.data(), this->buffer->ptr, size_bytes);
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
};

} // namespace NESO::Particles

#endif
