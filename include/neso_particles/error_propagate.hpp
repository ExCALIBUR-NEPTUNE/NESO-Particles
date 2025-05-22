#ifndef _NESO_PARTICLES_ERROR_PROPAGATE_H_
#define _NESO_PARTICLES_ERROR_PROPAGATE_H_

#include "compute_target.hpp"
#include "device_buffers.hpp"

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

  ~ErrorPropagate(){};

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

/**
 * Allow the NESO_KERNEL_ASSERT to be overridden to call a different function
 * with signature
 *
 * void(bool, int *);
 *
 */
#ifndef NESO_KERNEL_ASSERT_FUNCTION
/*
 *  Helper preprocessor macro to atomically increment the pointer in an
 *  ErrorPropagate class.
 */
#define NESO_KERNEL_ASSERT_FUNCTION(expr, ep_ptr)                              \
  if (!(expr)) {                                                               \
    atomic_fetch_add(ep_ptr, 1);                                               \
  }
#endif
#define NESO_KERNEL_ASSERT NESO_KERNEL_ASSERT_FUNCTION

using ErrorPropagateSharedPtr = std::shared_ptr<ErrorPropagate>;

/**
 * Helper function to create an ErrorPropagate instance.
 *
 * @param sycl_target SYCLTarget to use.
 * @returns New ErrorPropagate object.
 */
inline ErrorPropagateSharedPtr
error_propagate(SYCLTargetSharedPtr sycl_target) {
  return std::make_shared<ErrorPropagate>(sycl_target);
}

/**
 * ResourceStack interface for ErrorPropagate.
 */
struct ResourceStackInterfaceErrorPropagate : ResourceStackInterface<ErrorPropagate> {

  SYCLTargetSharedPtr sycl_target;
  ResourceStackInterfaceErrorPropagate(SYCLTargetSharedPtr sycl_target)
      : sycl_target(sycl_target) {}

  virtual inline std::shared_ptr<ErrorPropagate> construct() override {
    return std::make_shared<ErrorPropagate>(this->sycl_target);
  }

  virtual inline void
  free([[maybe_unused]] std::shared_ptr<ErrorPropagate> &resource) override {}

  virtual inline void
  clean([[maybe_unused]] std::shared_ptr<ErrorPropagate> &resource) override {
    resource->reset();
  }

};

/**
 * ResourceStackMap key for ResourceStackInterfaceErrorPropagate.
 */
struct ResourceStackKeyErrorPropagate {};

} // namespace NESO::Particles

#endif
