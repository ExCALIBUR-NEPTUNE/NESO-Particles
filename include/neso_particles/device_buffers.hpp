#ifndef _NESO_PARTICLES_DEVICE_BUFFERS_HPP_
#define _NESO_PARTICLES_DEVICE_BUFFERS_HPP_

#include "compute_target.hpp"

namespace NESO::Particles {

template <typename T> class BufferBase {
protected:
  virtual inline T *malloc_wrapper(const std::size_t num_bytes) = 0;
  virtual inline void free_wrapper(T *ptr) = 0;

  inline void generic_init() {
    NESOASSERT(this->ptr == nullptr, "Buffer is already allocated.");
    if (this->size > 0) {
      this->ptr = this->malloc_wrapper(this->size * sizeof(T));
      NESOASSERT(this->ptr != nullptr, "generic_init set nullptr.");
    }
  }
  inline void generic_free() {
    if (this->ptr != nullptr) {
      this->free_wrapper(this->ptr);
    }
    this->ptr = nullptr;
  }

  BufferBase(SYCLTargetSharedPtr sycl_target, std::size_t size)
      : sycl_target(sycl_target), ptr(nullptr), size(size) {}

  inline void assert_allocated() {
    NESOASSERT(this->ptr != nullptr,
               "Allocated buffer required but pointer is nullptr.");
  }

public:
  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// SYCL USM pointer to data.
  T *ptr;
  /// Number of elements allocated.
  std::size_t size;

  /**
   * Get the size of the allocation in bytes.
   *
   * @returns size of buffer in bytes.
   */
  inline std::size_t size_bytes() { return this->size * sizeof(T); }

  /**
   * Reallocate the buffer to hold at least the requested number of elements.
   * May or may not reduce the buffer size if called with a size less than the
   * current allocation. Current contents is not copied to the new buffer.
   *
   * @param size Minimum number of elements this buffer should be able to hold.
   * @param max_size_factor (Optional) Specify a ratio, if the underlying
   * buffer is larger than the requested ammount times this ratio then free the
   * buffer and reallocate.
   */
  inline int
  realloc_no_copy(const std::size_t size,
                  const std::optional<REAL> max_size_factor = std::nullopt) {

    const std::size_t max_size = std::max(
        size, (std::size_t)(max_size_factor != std::nullopt
                                ? max_size_factor.value() * ((REAL)size)
                                : this->size));

    if ((size > this->size) || (this->size > max_size)) {
      this->assert_allocated();
      this->free_wrapper(this->ptr);
      this->ptr = this->malloc_wrapper(size * sizeof(T));
      this->size = size;
    }
    return this->size;
  }

  /**
   * Asynchronously set the values in the buffer to those in a std::vector.
   *
   * @param data Input vector to copy values from.
   * @returns Event to wait on before using new values in the buffer.
   */
  inline sycl::event set_async(const std::vector<T> &data) {
    NESOASSERT(data.size() == this->size, "Input data is incorrectly sized.");
    const std::size_t size_bytes = sizeof(T) * this->size;
    if (size_bytes) {
      NESOASSERT(this->ptr != nullptr, "Internal pointer is not allocated.");
      this->assert_allocated();
      auto copy_event =
          sycl_target->queue.memcpy(this->ptr, data.data(), size_bytes);
      return copy_event;
    } else {
      return sycl::event();
    }
  }

  /**
   * Set the values in the buffer to those in a std::vector. Blocks until
   * the copy is complete.
   *
   * @param data Input vector to copy values from.
   */
  inline void set(const std::vector<T> &data) {
    this->set_async(data).wait_and_throw();
  }

  /**
   * Asynchronously get the values in the buffer into a std::vector.
   *
   * @param[in, out] data Input vector to copy values from buffer into.
   * @returns Event to wait on before using new values in the std::vector.
   */
  inline sycl::event get_async(std::vector<T> &data) {
    NESOASSERT(data.size() == this->size, "Input data is incorrectly sized.");
    const std::size_t size_bytes = sizeof(T) * this->size;
    if (size_bytes) {
      auto copy_event =
          sycl_target->queue.memcpy(data.data(), this->ptr, size_bytes);
      return copy_event;
    } else {
      return sycl::event();
    }
  }

  /**
   * Get the values in the buffer into a std::vector. Blocks until copy is
   * complete.
   *
   * @param[in, out] data Input vector to copy values from buffer into.
   */
  inline void get(std::vector<T> &data) {
    this->get_async(data).wait_and_throw();
  }

  /**
   * Get the values in the buffer into a std::vector.
   *
   * @returns std::vector of values in the buffer.
   */
  inline std::vector<T> get() {
    std::vector<T> data(this->size);
    this->get(data);
    return data;
  }
};

/**
 * Container around USM device allocated memory that can be resized.
 */
template <typename T> class BufferDevice : public BufferBase<T> {
protected:
  virtual inline T *malloc_wrapper(const std::size_t num_bytes) override {
    return static_cast<T *>(this->sycl_target->malloc_device(num_bytes));
  }
  virtual inline void free_wrapper(T *ptr) override {
    this->sycl_target->free(ptr);
  }

public:
  /// Disable (implicit) copies.
  BufferDevice(const BufferDevice &st) = delete;
  /// Disable (implicit) copies.
  BufferDevice &operator=(BufferDevice const &a) = delete;

  /**
   * Create a new BufferDevice of a given number of elements.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param size Number of elements.
   */
  BufferDevice(SYCLTargetSharedPtr &sycl_target, size_t size)
      : BufferBase<T>(sycl_target, size) {
    this->generic_init();
  }

  /**
   * Create a new BufferDevice from a std::vector. Note, this does not operate
   * like a sycl::buffer and is a copy of the source vector.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param vec Input vector to copy data from.
   */
  BufferDevice(SYCLTargetSharedPtr sycl_target, const std::vector<T> &vec)
      : BufferDevice(sycl_target, vec.size()) {
    this->set(vec);
  }

  virtual ~BufferDevice() { this->generic_free(); }
};

/**
 * Container around USM shared allocated memory that can be resized.
 */
template <typename T> class BufferShared : public BufferBase<T> {
protected:
  virtual inline T *malloc_wrapper(const std::size_t num_bytes) override {
    return static_cast<T *>(
        sycl::malloc_shared(num_bytes, this->sycl_target->queue));
  }
  virtual inline void free_wrapper(T *ptr) override {
    sycl::free(ptr, this->sycl_target->queue);
  }

public:
  /// Disable (implicit) copies.
  BufferShared(const BufferShared &st) = delete;
  /// Disable (implicit) copies.
  BufferShared &operator=(BufferShared const &a) = delete;

  /**
   * Create a new DeviceShared of a given number of elements.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param size Number of elements.
   */
  BufferShared(SYCLTargetSharedPtr sycl_target, size_t size)
      : BufferBase<T>(sycl_target, size) {
    this->generic_init();
  }

  /**
   * Create a new BufferShared from a std::vector. Note, this does not operate
   * like a sycl::buffer and is a copy of the source vector.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param vec Input vector to copy data from.
   */
  BufferShared(SYCLTargetSharedPtr sycl_target, const std::vector<T> &vec)
      : BufferShared(sycl_target, vec.size()) {
    this->set(vec);
  }

  ~BufferShared() { this->generic_free(); }
};

/**
 * Container around USM host allocated memory that can be resized.
 */
template <typename T> class BufferHost : public BufferBase<T> {
protected:
  virtual inline T *malloc_wrapper(const std::size_t num_bytes) override {
    return static_cast<T *>(this->sycl_target->malloc_host(num_bytes));
  }
  virtual inline void free_wrapper(T *ptr) override {
    this->sycl_target->free(ptr);
  }

public:
  /// Disable (implicit) copies.
  BufferHost(const BufferHost &st) = delete;
  /// Disable (implicit) copies.
  BufferHost &operator=(BufferHost const &a) = delete;

  /**
   * Create a new BufferHost of a given number of elements.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param size Number of elements.
   */
  BufferHost(SYCLTargetSharedPtr sycl_target, size_t size)
      : BufferBase<T>(sycl_target, size) {
    this->generic_init();
  }

  /**
   * Create a new BufferHost from a std::vector. Note, this does not operate
   * like a sycl::buffer and is a copy of the source vector.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param vec Input vector to copy data from.
   */
  BufferHost(SYCLTargetSharedPtr sycl_target, const std::vector<T> &vec)
      : BufferHost(sycl_target, vec.size()) {
    this->set(vec);
  }

  ~BufferHost() { this->generic_free(); }
};

/**
 * Wrapper around a BufferDevice and a BufferHost to store data on the device
 * and the host. To be used as an alternative to BufferShared where the
 * programmer wants to explicitly handle when data is copied between the device
 * and the host.
 */
template <typename T> class BufferDeviceHost {
private:
public:
  /// Disable (implicit) copies.
  BufferDeviceHost(const BufferDeviceHost &st) = delete;
  /// Disable (implicit) copies.
  BufferDeviceHost &operator=(BufferDeviceHost const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// Size of the allocation on device and host.
  size_t size;

  /// Wrapped BufferDevice.
  BufferDevice<T> d_buffer;
  /// Wrapped BufferHost.
  BufferHost<T> h_buffer;

  ~BufferDeviceHost(){};

  /**
   * Create a new BufferDeviceHost of the request size on the requested compute
   * target.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param size Number of elements to initially allocate on the device and
   * host.
   */
  BufferDeviceHost(SYCLTargetSharedPtr sycl_target, size_t size)
      : sycl_target(sycl_target), size(size), d_buffer(sycl_target, size),
        h_buffer(sycl_target, size){};

  /**
   * Create a new BufferDeviceHost from a std::vector. Note, this does not
   * operate like a sycl::buffer and is a copy of the source vector.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param vec Input vector to copy data from.
   */
  BufferDeviceHost(SYCLTargetSharedPtr sycl_target, const std::vector<T> &vec)
      : sycl_target(sycl_target), size(vec.size()), d_buffer(sycl_target, vec),
        h_buffer(sycl_target, vec){};

  /**
   * Get the size in bytes of the allocation on the host and device.
   *
   * @returns Number of bytes allocated on the host and device.
   */
  inline size_t size_bytes() { return this->size * sizeof(T); }

  /**
   * Reallocate both the device and host buffers to hold at least the requested
   * number of elements. May or may not reduce the buffer size if called with a
   * size less than the current allocation. Current contents is not copied to
   * the new buffer.
   *
   * @param size Minimum number of elements this buffer should be able to hold.
   */
  inline int realloc_no_copy(const size_t size) {
    this->d_buffer.realloc_no_copy(size);
    this->h_buffer.realloc_no_copy(size);
    this->size = size;
    return this->size;
  }

  /**
   * Copy the contents of the host buffer to the device buffer.
   */
  inline void host_to_device() {
    if (this->size_bytes() > 0) {
      this->sycl_target->queue
          .memcpy(this->d_buffer.ptr, this->h_buffer.ptr, this->size_bytes())
          .wait();
    }
  }
  /**
   * Copy the contents of the device buffer to the host buffer.
   */
  inline void device_to_host() {
    if (this->size_bytes() > 0) {
      this->sycl_target->queue
          .memcpy(this->h_buffer.ptr, this->d_buffer.ptr, this->size_bytes())
          .wait();
    }
  }
  /**
   * Start an asynchronous copy of the host data to the device buffer.
   *
   * @returns sycl::event to wait on for completion of data movement.
   */
  inline sycl::event async_host_to_device() {
    NESOASSERT(this->size_bytes() > 0, "Zero sized copy issued.");
    return this->sycl_target->queue.memcpy(
        this->d_buffer.ptr, this->h_buffer.ptr, this->size_bytes());
  }
  /**
   * Start an asynchronous copy of the device data to the host buffer.
   *
   * @returns sycl::event to wait on for completion of data movement.
   */
  inline sycl::event async_device_to_host() {
    NESOASSERT(this->size_bytes() > 0, "Zero sized copy issued.");
    return this->sycl_target->queue.memcpy(
        this->h_buffer.ptr, this->d_buffer.ptr, this->size_bytes());
  }
};

/**
 * Copy the contents of a buffer, e.g. BufferDevice, BufferHost, into another
 * buffer.
 *
 * @param dst Destination buffer.
 * @param src Source buffer.
 */
template <template <typename> typename D, template <typename> typename S,
          typename T>
[[nodiscard]] inline sycl::event buffer_memcpy(D<T> &dst, S<T> &src) {
  NESOASSERT(dst.size == src.size, "Buffers have different sizes.");
  return dst.sycl_target->queue.memcpy(dst.ptr, src.ptr, dst.size * sizeof(T));
}
} // namespace NESO::Particles
#endif
