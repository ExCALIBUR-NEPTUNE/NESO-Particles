#ifndef _NESO_PARTICLES_ND_LOCAL_ARRAY_H_
#define _NESO_PARTICLES_ND_LOCAL_ARRAY_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "nd_index.hpp"

namespace NESO::Particles {

/**
 * Generic N-Dimensional array type which is accessible on the host and in a
 * @ref ParticleLoop kernel.
 */
template <typename T, std::size_t N> class NDLocalArray {
protected:
  std::unique_ptr<BufferDevice<T>> buffer;
  INT size;

public:
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
    this->index = {shape...};
    this->size = this->index.size();
    this->buffer =
        std::make_unique<BufferDevice<T>>(this->sycl_target, this->size);
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
