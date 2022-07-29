#ifndef _NESO_PARTICLES_COMPUTE_TARGET
#define _NESO_PARTICLES_COMPUTE_TARGET

#include <CL/sycl.hpp>
#include <mpi.h>
#include <stack>

#include "communication.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;

namespace NESO::Particles {

/**
 * Container for SYCL devices and queues such that they can be easily passed
 * around.
 */
class SYCLTarget {
private:
public:
  /// SYCL device in use.
  sycl::device device;
  /// Main SYCL queue to use.
  sycl::queue queue;
  /// Parent MPI communicator to use.
  MPI_Comm comm;
  /// CommPair to pass around inter and intra communicators for shared memory
  /// regions.
  CommPair comm_pair;
  /// ProfileMap to log profiling data related to this SYCLTarget.
  ProfileMap profile_map;

  SYCLTarget(){};

  /**
   * Create a new SYCLTarget using a flag to specifiy device type and a parent
   * MPI communicator.
   *
   * @param gpu_device Specify how to search for SYCL device. 0 uses default
   * selector, 1 explicitly use a GPU selector and -1 use an explicit CPU
   * selector.
   * @param sycl_target SYCLTarget to use as compute device.
   */
  SYCLTarget(const int gpu_device, MPI_Comm comm) : comm_pair(comm) {
    if (gpu_device > 0) {
      try {
        this->device = sycl::device(sycl::gpu_selector());
      } catch (sycl::exception const &e) {
        std::cout << "Cannot select a GPU\n" << e.what() << "\n";
        std::cout << "Using a CPU device\n";
        this->device = sycl::device(sycl::cpu_selector());
      }
    } else if (gpu_device < 0) {
      this->device = sycl::device(sycl::cpu_selector());
    } else {
      this->device = sycl::device(sycl::default_selector());
    }

    if (this->comm_pair.rank_parent == 0) {
      std::cout << "Using " << this->device.get_info<sycl::info::device::name>()
                << std::endl;
      std::cout << "Kernel type: " << NESO_PARTICLES_DEVICE_LABEL << std::endl;
    }

    this->queue = sycl::queue(this->device);
    this->comm = comm;

    this->profile_map.set("MPI", "MPI_COMM_WORLD_rank",
                          this->comm_pair.rank_parent);
    this->profile_map.set("MPI", "MPI_COMM_WORLD_size",
                          this->comm_pair.size_parent);
  }
  ~SYCLTarget() {}

  /**
   * Free the SYCLTarget and underlying CommPair.
   */
  void free() { comm_pair.free(); }
};

/**
 * Container around USM device allocated memory that can be resized.
 */
template <typename T> class BufferDevice {
private:
public:
  /// Compute device used by the instance.
  SYCLTarget &sycl_target;
  /// SYCL USM device pointer, only accessible on device.
  T *ptr;
  /// Number of elements allocated.
  size_t size;

  /**
   * Create a new DeviceBuffer of a given number of elements.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param size Number of elements.
   */
  BufferDevice(SYCLTarget &sycl_target, size_t size)
      : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl::malloc_device(size * sizeof(T), sycl_target.queue);
  }
  /**
   * Get the size of the allocation in bytes.
   *
   * @returns size of buffer in bytes.
   */
  inline size_t size_bytes() { return this->size * sizeof(T); }

  /**
   * Reallocate the buffer to hold at least the requested number of elements.
   * May or may not reduce the buffer size if called with a size less than the
   * current allocation. Current contents is not copied to the new buffer.
   *
   * @param size Minimum number of elements this buffer should be able to hold.
   */
  inline int realloc_no_copy(const size_t size) {
    if (size > this->size) {
      sycl::free(this->ptr, this->sycl_target.queue);
      this->ptr = (T *)sycl::malloc_device(size * sizeof(T), sycl_target.queue);
      this->size = size;
    }
    return this->size;
  }
  ~BufferDevice() {
    if (this->ptr != NULL) {
      sycl::free(this->ptr, sycl_target.queue);
    }
  }
};

/**
 * Container around USM shared allocated memory that can be resized.
 */
template <typename T> class BufferShared {
private:
public:
  /// Compute device used by the instance.
  SYCLTarget &sycl_target;
  /// SYCL USM shared pointer, accessible on host and device.
  T *ptr;
  /// Number of elements allocated.
  size_t size;

  /**
   * Create a new DeviceShared of a given number of elements.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param size Number of elements.
   */
  BufferShared(SYCLTarget &sycl_target, size_t size)
      : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl::malloc_shared(size * sizeof(T), sycl_target.queue);
  }
  /**
   * Get the size of the allocation in bytes.
   *
   * @returns size of buffer in bytes.
   */
  inline size_t size_bytes() { return this->size * sizeof(T); }
  /**
   * Reallocate the buffer to hold at least the requested number of elements.
   * May or may not reduce the buffer size if called with a size less than the
   * current allocation. Current contents is not copied to the new buffer.
   *
   * @param size Minimum number of elements this buffer should be able to hold.
   */
  inline int realloc_no_copy(const size_t size) {
    if (size > this->size) {
      sycl::free(this->ptr, this->sycl_target.queue);
      this->ptr = (T *)sycl::malloc_shared(size * sizeof(T), sycl_target.queue);
      this->size = size;
    }
    return this->size;
  }
  ~BufferShared() {
    if (this->ptr != NULL) {
      sycl::free(this->ptr, sycl_target.queue);
    }
  }
};

/**
 * Container around USM host allocated memory that can be resized.
 */
template <typename T> class BufferHost {
private:
public:
  /// Compute device used by the instance.
  SYCLTarget &sycl_target;
  /// SYCL USM shared pointer, accessible on host and device.
  T *ptr;
  /// Number of elements allocated.
  size_t size;

  /**
   * Create a new DeviceHost of a given number of elements.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param size Number of elements.
   */
  BufferHost(SYCLTarget &sycl_target, size_t size) : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl::malloc_host(size * sizeof(T), sycl_target.queue);
  }

  /**
   * Get the size of the allocation in bytes.
   *
   * @returns size of buffer in bytes.
   */
  inline size_t size_bytes() { return this->size * sizeof(T); }

  /**
   * Reallocate the buffer to hold at least the requested number of elements.
   * May or may not reduce the buffer size if called with a size less than the
   * current allocation. Current contents is not copied to the new buffer.
   *
   * @param size Minimum number of elements this buffer should be able to hold.
   */
  inline int realloc_no_copy(const size_t size) {
    if (size > this->size) {
      sycl::free(this->ptr, this->sycl_target.queue);
      this->ptr = (T *)sycl::malloc_host(size * sizeof(T), sycl_target.queue);
      this->size = size;
    }
    return this->size;
  }
  ~BufferHost() {
    if (this->ptr != NULL) {
      sycl::free(this->ptr, sycl_target.queue);
    }
  }
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
  /// Compute device used by the instance.
  SYCLTarget &sycl_target;
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
   * @param sycl_target SYCLTarget to use as compute device.
   * @param size Number of elements to initially allocate on the device and
   * host.
   */
  BufferDeviceHost(SYCLTarget &sycl_target, size_t size)
      : sycl_target(sycl_target), size(size), d_buffer(sycl_target, size),
        h_buffer(sycl_target, size){};

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
    this->sycl_target.queue
        .memcpy(this->d_buffer.ptr, this->h_buffer.ptr, this->size_bytes())
        .wait();
  }
  /**
   * Copy the contents of the device buffer to the host buffer.
   */
  inline void device_to_host() {
    this->sycl_target.queue
        .memcpy(this->h_buffer.ptr, this->d_buffer.ptr, this->size_bytes())
        .wait();
  }
  /**
   * Start an asynchronous copy of the host data to the device buffer.
   *
   * @returns sycl::event to wait on for completion of data movement.
   */
  inline sycl::event async_host_to_device() {
    return this->sycl_target.queue.memcpy(
        this->d_buffer.ptr, this->h_buffer.ptr, this->size_bytes());
  }
  /**
   * Start an asynchronous copy of the device data to the host buffer.
   *
   * @returns sycl::event to wait on for completion of data movement.
   */
  inline sycl::event async_device_to_host() {
    return this->sycl_target.queue.memcpy(
        this->h_buffer.ptr, this->d_buffer.ptr, this->size_bytes());
  }
};

/**
 * Helper class to hold a collection of sycl::event instances which can be
 * waited on.
 */
class EventStack {
private:
  std::stack<sycl::event> stack;

public:
  ~EventStack(){};

  /**
   *  Create a new and empty stack of events.
   */
  EventStack(){};

  /**
   *  Push a sycl::event onto the event stack.
   */
  inline void push(sycl::event e) { this->stack.push(e); }

  /**
   *  Wait for all events held in the stack to complete before returning.
   */
  inline void wait() {
    while (!this->stack.empty()) {
      this->stack.top().wait();
      this->stack.pop();
    }
  };
};

} // namespace NESO::Particles

#endif
