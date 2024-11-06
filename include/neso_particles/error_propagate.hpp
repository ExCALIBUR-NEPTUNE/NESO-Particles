#ifndef _NESO_PARTICLES_ERROR_PROPAGATE_H_
#define _NESO_PARTICLES_ERROR_PROPAGATE_H_

#include "compute_target.hpp"

namespace NESO::Particles {

/**
 *  Helper class to propagate errors thrown from inside a SYCL kernel
 */
class ErrorPropagate {
private:
  SYCLTargetSharedPtr sycl_target;
  BufferDeviceHost<int> dh_flag;

public:
  /// Disable (implicit) copies.
  ErrorPropagate(const ErrorPropagate &st) = delete;
  /// Disable (implicit) copies.
  ErrorPropagate &operator=(ErrorPropagate const &a) = delete;

  ~ErrorPropagate() {};

  /**
   * Create a new instance to track assertions thrown in SYCL kernels.
   *
   * @param sycl_target SYCLTargetSharedPtr used for the kernel.
   */
  ErrorPropagate(SYCLTargetSharedPtr sycl_target)
      : sycl_target(sycl_target), dh_flag(sycl_target, 1) {
    this->reset();
  }

  /**
   *  Reset the internal state. Useful if the instance is used to indicate
   *  events occurred in a parallel loop that are non fatal.
   */
  inline void reset() {
    this->dh_flag.h_buffer.ptr[0] = 0;
    this->dh_flag.host_to_device();
  }

  /**
   * Get the int device pointer for use in a SYCL kernel. This pointer should
   * be incremented atomically with some positive integer to indicate an error
   * occured in the kernel.
   */
  inline int *device_ptr() { return this->dh_flag.d_buffer.ptr; }

  /**
   * Check the stored integer. If the integer is non-zero throw an error with
   * the passed message.
   *
   * @param error_msg Message to throw if stored integer is non-zero.
   */
  inline void check_and_throw(const char *error_msg) {
    NESOASSERT(this->get_flag() == 0, error_msg);
  }

  /**
   *  Get the current value of the flag on the host.
   *
   *  @returns flag.
   */
  inline int get_flag() {
    this->dh_flag.device_to_host();
    return this->dh_flag.h_buffer.ptr[0];
  }
};

/*
 *  Helper preprocessor macro to atomically increment the pointer in an
 *  ErrorPropagate class.
 */
#define NESO_KERNEL_ASSERT(expr, ep_ptr)                                       \
  if (!(expr)) {                                                               \
    sycl::atomic_ref<int, sycl::memory_order::relaxed,                         \
                     sycl::memory_scope::device>                               \
        neso_error_atomic(ep_ptr[0]);                                          \
    neso_error_atomic.fetch_add(1);                                            \
  }

} // namespace NESO::Particles

#endif
