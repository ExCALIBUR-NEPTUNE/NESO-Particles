#ifndef _NESO_PARTICLES_HOST_RNG_COMMON_H_
#define _NESO_PARTICLES_HOST_RNG_COMMON_H_

#include "../../compute_target.hpp"
#include <functional>
#include <tuple>

namespace NESO::Particles {

/**
 * Draw values from a host generation function and copy them to a device
 * buffer.
 *
 * @param sycl_target Compute device.
 * @param func Generation function to call for each value.
 * @param d_ptr Pointer to buffer in which to place values.
 * @param num_numbers Number of values to draw from generation function.
 * @param block_size Block size to use when copying into device buffer.
 */
template <typename FUNC_TYPE, typename T>
inline void draw_random_samples(SYCLTargetSharedPtr sycl_target, FUNC_TYPE func,
                                T *d_ptr, const std::size_t num_numbers,
                                const int block_size) {
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
    const std::size_t num_to_memcpy = std::min(
        static_cast<std::size_t>(block_size), num_numbers - num_numbers_moved);
    for (std::size_t ix = 0; ix < num_to_memcpy; ix++) {
      ptr_current[ix] = func();
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

  inline T *allocate(SYCLTargetSharedPtr sycl_target, const int nrow,
                     bool *reallocated = nullptr) {
    auto size_start = this->get_buffer_size(sycl_target);

    if (nrow <= 0) {
      return nullptr;
    }
    const std::size_t required_size =
        static_cast<std::size_t>(nrow) *
        static_cast<std::size_t>(this->num_components);

    if (!this->d_buffers.count(sycl_target)) {
      this->d_buffers[sycl_target] =
          std::make_unique<BufferDevice<T>>(sycl_target, required_size);
    } else {
      this->d_buffers.at(sycl_target)->realloc_no_copy(required_size, 1.2);
    }

    auto size_end = this->get_buffer_size(sycl_target);
    if (reallocated != nullptr) {
      *reallocated = size_start != size_end;
    }
    return this->d_buffers.at(sycl_target)->ptr;
  }

  inline T *get_buffer_ptr(SYCLTargetSharedPtr sycl_target) {
    return this->d_buffers.at(sycl_target)->ptr;
  }

public:
  /// The number of RNG values required per particle.
  int num_components;
  /// RNG values are sampled and copied to the device in this block size.
  int block_size;

  BlockKernelRNGBase(const int num_components, const int block_size = 8192)
      : num_components(num_components), block_size(block_size) {}
};

} // namespace NESO::Particles

#endif
