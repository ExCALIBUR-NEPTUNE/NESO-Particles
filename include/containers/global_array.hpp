#ifndef _NESO_PARTICLES_GLOBAL_ARRAY_H_
#define _NESO_PARTICLES_GLOBAL_ARRAY_H_

#include "../compute_target.hpp"
#include "../typedefs.hpp"

#include "../communication.hpp"
#include <memory>
#include <mpi.h>
#include <optional>
#include <vector>

namespace NESO::Particles {

// Forward declaration of ParticleLoop such that GlobalArray can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;

/**
 *  GlobalArray is an array type which can be accessed from kernels in read or
 *  atomic add mode. Post loop execution, with add access mode, the global
 *  array values are automatically reduced across the MPI communicator.
 */
template <typename T> class GlobalArray {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;

protected:
  std::shared_ptr<BufferDeviceHost<T>> buffer;

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T *impl_get() { return this->buffer->d_buffer.ptr; }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop.
   */
  inline T const *impl_get_const() { return this->buffer->d_buffer.ptr; }

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
