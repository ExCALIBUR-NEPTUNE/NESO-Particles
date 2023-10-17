#ifndef _NESO_PARTICLES_LOCAL_ARRAY_H_
#define _NESO_PARTICLES_LOCAL_ARRAY_H_

#include "../compute_target.hpp"

namespace NESO::Particles {

/**
 * Container to hold an array of values on each MPI rank.
 */
template <typename T> class LocalArray {
protected:
  std::shared_ptr<BufferDevice<T>> buffer;

public:
  /// The SYCLTarget the LocalArray is created on.
  SYCLTargetSharedPtr sycl_target;
  /// The number of elements in the array.
  std::size_t size;

  LocalArray() = default;
  LocalArray<T> &operator=(const LocalArray<T> &) = default;

  /**
   * TODO
   */
  inline T *impl_get() { return this->buffer->ptr; }

  /**
   * TODO
   */
  inline T const *impl_get_const() { return this->buffer->ptr; }

  /**
   *  TODO
   */
  LocalArray(SYCLTargetSharedPtr sycl_target, const std::size_t size)
      : sycl_target(sycl_target), size(size) {
    this->buffer = std::make_shared<BufferDevice<T>>(sycl_target, size);
  }

  /**
   *  TODO
   */
  LocalArray(SYCLTargetSharedPtr sycl_target, const std::vector<T> &data)
      : LocalArray(sycl_target, data.size()) {
    this->set(data);
  }

  /**
   * TODO
   */
  inline sycl::event set_async(const std::vector<T> &data) {
    NESOASSERT(data.size() == this->size, "Input data is incorrectly sized.");
    sycl::buffer<T> b_data(data.data(), sycl::range<1>(data.size()));
    T *ptr = this->buffer->ptr;
    sycl::event copy_event =
        this->sycl_target->queue.submit([&](sycl::handler &cgh) {
          sycl::accessor a_data{b_data, cgh, sycl::read_only};
          cgh.copy(a_data, ptr);
        });
    return copy_event;
  }

  /**
   * TODO
   */
  inline void set(const std::vector<T> &data) {
    this->set_async(data).wait_and_throw();
  }

  /**
   * TODO
   */
  inline sycl::event get_async(std::vector<T> &data) {
    NESOASSERT(data.size() == this->size, "Input data is incorrectly sized.");
    sycl::buffer<T> b_data(data.data(), sycl::range<1>(data.size()));
    T *ptr = this->buffer->ptr;
    sycl::event copy_event =
        this->sycl_target->queue.submit([&](sycl::handler &cgh) {
          sycl::accessor a_data{b_data, cgh, sycl::write_only, sycl::no_init};
          cgh.copy(ptr, a_data);
        });
    return copy_event;
  }

  /**
   * TODO
   */
  inline void get(std::vector<T> &data) {
    this->get_async(data).wait_and_throw();
  }
};

} // namespace NESO::Particles

#endif
