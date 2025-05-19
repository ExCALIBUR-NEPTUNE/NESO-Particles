#ifndef _NESO_PARTICLES_HOST_RNG_COMMON_H_
#define _NESO_PARTICLES_HOST_RNG_COMMON_H_

#include "../../compute_target.hpp"
#include "../../device_buffers.hpp"
#include <functional>
#include <tuple>

namespace NESO::Particles {

/**
 * This type defines the absract interface for creating RNG values in a block.
 */
template <typename T> struct RNGGenerationFunction {
  virtual ~RNGGenerationFunction() = default;

  /**
   * Draw values from a host generation function and copy them to a device
   * buffer.
   *
   * @param[in] sycl_target Compute device.
   * @param[in, out] d_ptr Pointer to buffer in which to place values.
   * @param[in] num_numbers Number of values to draw from generation function.
   * @param[in] block_size Block size to use when copying into device buffer.
   */
  virtual inline void draw_random_samples(SYCLTargetSharedPtr sycl_target,
                                          T *d_ptr,
                                          const std::size_t num_numbers,
                                          const int block_size) = 0;
};

template <typename T>
struct HostRNGGenerationFunction : RNGGenerationFunction<T> {
  virtual ~HostRNGGenerationFunction() = default;

  /// The host callable that returns RNG samples.
  std::function<T()> generation_function;

  /**
   * @param generation_function Host callable function that returns a sample on
   * each call.
   */
  HostRNGGenerationFunction(std::function<T()> generation_function)
      : generation_function(generation_function) {}

  /**
   * Draw values from a host generation function and copy them to a device
   * buffer.
   *
   * @param[in] sycl_target Compute device.
   * @param[in, out] d_ptr Pointer to buffer in which to place values.
   * @param[in] num_numbers Number of values to draw from generation function.
   * @param[in] block_size Block size to use when copying into device buffer.
   */
  virtual inline void draw_random_samples(SYCLTargetSharedPtr sycl_target,
                                          T *d_ptr,
                                          const std::size_t num_numbers,
                                          const int block_size) override {
    auto d_ptr_start = d_ptr;

    // Create the random number in blocks and copy to device blockwise.
    std::vector<T> block0(block_size);
    std::vector<T> block1(block_size);

    T *ptr_tmp;
    T *ptr_current = block0.data();
    T *ptr_next = block1.data();
    std::size_t num_numbers_moved = 0;

    sycl::event e;
    while (num_numbers_moved < num_numbers) {

      // Create a block of samples
      const std::size_t num_to_memcpy =
          std::min(static_cast<std::size_t>(block_size),
                   num_numbers - num_numbers_moved);
      for (std::size_t ix = 0; ix < num_to_memcpy; ix++) {
        ptr_current[ix] = this->generation_function();
      }

      // Wait until the previous block finished copying before starting this
      // copy
      e.wait_and_throw();
      e = sycl_target->queue.memcpy(d_ptr, ptr_current,
                                    num_to_memcpy * sizeof(T));
      d_ptr += num_to_memcpy;
      num_numbers_moved += num_to_memcpy;

      // swap ptr_current and ptr_next such that the new samples are written
      // into ptr_next whilst ptr_current is being copied to the device.
      ptr_tmp = ptr_current;
      ptr_current = ptr_next;
      ptr_next = ptr_tmp;
    }

    e.wait_and_throw();

    NESOASSERT(num_numbers_moved == num_numbers,
               "Failed to copy the correct number of random numbers");
    NESOASSERT(d_ptr == d_ptr_start + num_numbers,
               "Failed to copy the correct number of random numbers (pointer "
               "arithmetic)");
  }
};

template <typename T> class BlockKernelRNGBase {
protected:
  std::map<SYCLTargetSharedPtr, std::shared_ptr<BufferDevice<T>>> d_buffers;

  inline std::size_t get_buffer_size(SYCLTargetSharedPtr sycl_target) {
    if (!this->d_buffers.count(sycl_target)) {
      return 0;
    } else {
      return this->d_buffers.at(sycl_target)->size;
    }
  }

  inline T *allocate(SYCLTargetSharedPtr sycl_target,
                     const std::size_t required_size,
                     bool *reallocated = nullptr) {
    if (required_size == 0) {
      return nullptr;
    }

    const std::size_t padded_size = std::max(
        required_size,
        static_cast<std::size_t>(static_cast<REAL>(required_size) * 1.1));

    if (!this->d_buffers.count(sycl_target)) {
      this->d_buffers[sycl_target] =
          std::make_unique<BufferDevice<T>>(sycl_target, padded_size);
    } else {
      if (this->d_buffers.at(sycl_target)->size < required_size) {
        this->d_buffers.at(sycl_target)->realloc(padded_size, this->max_factor);
      }
    }

    if (reallocated != nullptr) {
      *reallocated = false;
    }
    return this->d_buffers.at(sycl_target)->ptr;
  }

  inline T *get_buffer_ptr(SYCLTargetSharedPtr sycl_target) {
    return this->d_buffers.at(sycl_target)->ptr;
  }

public:
  /// Implementation to generate samples.
  std::shared_ptr<RNGGenerationFunction<T>> generation_function;
  /// The number of RNG values required per particle.
  int num_components;
  /// RNG values are sampled and copied to the device in this block size.
  int block_size;
  /// Factor for allocation
  std::optional<REAL> max_factor;

  BlockKernelRNGBase()
      : generation_function(nullptr), num_components(0), block_size(8192),
        max_factor(std::nullopt) {}

  BlockKernelRNGBase(
      std::shared_ptr<RNGGenerationFunction<T>> generation_function,
      const int num_components, const int block_size = 8192)
      : generation_function(generation_function),
        num_components(num_components), block_size(block_size),
        max_factor(std::nullopt) {}
};

} // namespace NESO::Particles

#endif
