#ifndef _NESO_PARTICLES_COMPUTE_TARGET
#define _NESO_PARTICLES_COMPUTE_TARGET

#include <cstdlib>

#include <CL/sycl.hpp>
#include <array>
#include <map>
#include <mpi.h>
#include <stack>
#include <string>
#include <vector>

#include "communication.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;

namespace NESO::Particles {

/**
 *  Determine a local MPI rank based on environment variables and shared memory
 *  splitting.
 *
 *  @param comm  MPI_Comm to try and deduce local rank from.
 *  @param default_rank (optional)  MPI rank to use if one cannot be determined
 *  from environment variables or SHM intra comm.
 */
inline int get_local_mpi_rank(MPI_Comm comm, int default_rank = -1) {

  if (const char *env_char = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
    std::string env_str = std::string(env_char);
    const int env_int = std::stoi(env_str);
    return env_int;
  } else if (const char *env_char = std::getenv("MV2_COMM_WORLD_LOCAL_RANK")) {
    std::string env_str = std::string(env_char);
    const int env_int = std::stoi(env_str);
    return env_int;
  } else if (const char *env_char = std::getenv("MPI_LOCALRANKID")) {
    std::string env_str = std::string(env_char);
    const int env_int = std::stoi(env_str);
    return env_int;
  } else if (default_rank < 0) {
    CommPair comm_pair(comm);
    default_rank = comm_pair.rank_intra;
    comm_pair.free();
  }

  return default_rank;
}

/**
 * Container for SYCL devices and queues such that they can be easily passed
 * around.
 */
class SYCLTarget {
private:
  std::map<unsigned char *, size_t> ptr_map;

#ifdef DEBUG_OOB_CHECK
  std::array<unsigned char, DEBUG_OOB_WIDTH> ptr_bit_mask;
  std::array<unsigned char, DEBUG_OOB_WIDTH> ptr_bit_tmp;
#endif

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

  /// Disable (implicit) copies.
  SYCLTarget(const SYCLTarget &st) = delete;
  /// Disable (implicit) copies.
  SYCLTarget &operator=(SYCLTarget const &a) = delete;

  /**
   * Create a new SYCLTarget using a flag to specifiy device type and a parent
   * MPI communicator.
   *
   * @param gpu_device Specify how to search for SYCL device. 0 uses default
   * selector, 1 explicitly use a GPU selector and -1 use an explicit CPU
   * selector.
   * @param comm MPI Communicator on which the SYCLTarget is based.
   * @param local_rank (optional) Explicitly pass the local rank used to assign
   * devices to MPI ranks.
   */
  SYCLTarget(const int gpu_device, MPI_Comm comm, int local_rank = -1)
      : comm_pair(comm) {
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

      // Get the default device and platform as they are most likely to be the
      // desired device based on SYCL implementation/runtime/environment
      // variables.
      auto default_device = sycl::device(sycl::default_selector());
      auto default_platform = default_device.get_platform();

      // Get all devices from the default platform
      auto devices = default_platform.get_devices();

      // determine the local rank to use for round robin device assignment.
      if (local_rank < 0) {
        local_rank = get_local_mpi_rank(comm, this->comm_pair.rank_intra);
      }

      // round robin assign devices to local MPI ranks.
      const int num_devices = devices.size();
      const int device_index = local_rank % num_devices;
      this->device = devices[device_index];

      this->profile_map.set("MPI", "MPI_COMM_WORLD_rank_local", local_rank);
      this->profile_map.set("SYCL", "DEVICE_COUNT", num_devices);
      this->profile_map.set("SYCL", "DEVICE_INDEX", device_index);
      this->profile_map.set(
          "SYCL", this->device.get_info<sycl::info::device::name>(), 0);
    }

    this->queue = sycl::queue(this->device);
    this->comm = comm;

    this->profile_map.set("MPI", "MPI_COMM_WORLD_rank",
                          this->comm_pair.rank_parent);
    this->profile_map.set("MPI", "MPI_COMM_WORLD_size",
                          this->comm_pair.size_parent);

#ifdef DEBUG_OOB_CHECK
    for (int cx = 0; cx < DEBUG_OOB_WIDTH; cx++) {
      this->ptr_bit_mask[cx] = static_cast<unsigned char>(255);
    }
#endif
  }
  ~SYCLTarget() {
#ifdef DEBUG_OOB_CHECK
    if (this->ptr_map.size() > 0) {
      for (auto &px : this->ptr_map) {
        nprint("NOT FREED:", (void *)px.first, px.second);
      }
    } else {
      nprint("ALL FREED");
    }
#endif
  }

  /**
   * Print information to stdout about the current SYCL device (on MPI rank 0).
   */
  inline void print_device_info() {
    if (this->comm_pair.rank_parent == 0) {
      std::cout << "Using " << this->device.get_info<sycl::info::device::name>()
                << std::endl;
      std::cout << "Kernel type: " << NESO_PARTICLES_DEVICE_LABEL << std::endl;
    }
  }

  /**
   * Free the SYCLTarget and underlying CommPair.
   */
  inline void free() { comm_pair.free(); }

  /**
   * Allocate memory on device using sycl::malloc_device.
   *
   * @param size_bytes Number of bytes to allocate.
   */
  inline void *malloc_device(size_t size_bytes) {

#ifndef DEBUG_OOB_CHECK
    return sycl::malloc_device(size_bytes, this->queue);
#else

    unsigned char *ptr = (unsigned char *)sycl::malloc_device(
        size_bytes + 2 * DEBUG_OOB_WIDTH, this->queue);
    unsigned char *ptr_user = ptr + DEBUG_OOB_WIDTH;
    this->ptr_map[ptr_user] = size_bytes;
    NESOASSERT(ptr != nullptr, "pad pointer from malloc_device");

    this->queue.memcpy(ptr, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH).wait();

    this->queue
        .memcpy(ptr_user + size_bytes, this->ptr_bit_mask.data(),
                DEBUG_OOB_WIDTH)
        .wait();

    return (void *)ptr_user;
    // return ptr;
#endif
  }

  /**
   * Allocate memory on the host using sycl::malloc_host.
   *
   * @param size_bytes Number of bytes to allocate.
   */
  inline void *malloc_host(size_t size_bytes) {

#ifndef DEBUG_OOB_CHECK
    return sycl::malloc_host(size_bytes, this->queue);
#else

    unsigned char *ptr = (unsigned char *)sycl::malloc_host(
        size_bytes + 2 * DEBUG_OOB_WIDTH, this->queue);
    unsigned char *ptr_user = ptr + DEBUG_OOB_WIDTH;
    this->ptr_map[ptr_user] = size_bytes;
    NESOASSERT(ptr != nullptr, "pad pointer from malloc_host");

    this->queue.memcpy(ptr, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH).wait();

    this->queue
        .memcpy(ptr_user + size_bytes, this->ptr_bit_mask.data(),
                DEBUG_OOB_WIDTH)
        .wait();

    return (void *)ptr_user;
    // return ptr;
#endif
  }

  /**
   *  Free a pointer allocated with malloc_device.
   *
   *  @param ptr_in Pointer to free.
   */
  template <typename T> inline void free(T *ptr_in) {
#ifndef DEBUG_OOB_CHECK
    sycl::free(ptr_in, this->queue);
#else
    unsigned char *ptr = (unsigned char *)ptr_in;
    NESOASSERT(this->ptr_map.count(ptr), "point not alloced correctly");

    this->check_ptr(ptr, this->ptr_map[ptr]);

    this->ptr_map.erase(ptr);

    sycl::free((void *)(ptr - DEBUG_OOB_WIDTH), this->queue);
    // sycl::free(ptr_in, this->queue);
#endif
  }

  inline void check_ptrs() {
    for (auto &px : this->ptr_map) {
      this->check_ptr(px.first, px.second);
    }
  }

  inline void check_ptr(unsigned char *ptr_user, const size_t size_bytes) {

#ifdef DEBUG_OOB_CHECK
    this->queue
        .memcpy(this->ptr_bit_tmp.data(), ptr_user - DEBUG_OOB_WIDTH,
                DEBUG_OOB_WIDTH)
        .wait();

    for (int cx = 0; cx < DEBUG_OOB_WIDTH; cx++) {
      NESOASSERT(this->ptr_bit_tmp[cx] == static_cast<unsigned char>(255),
                 "DEBUG PADDING START TOUCHED");
    }

    this->queue
        .memcpy(this->ptr_bit_tmp.data(), ptr_user + size_bytes,
                DEBUG_OOB_WIDTH)
        .wait();

    for (int cx = 0; cx < DEBUG_OOB_WIDTH; cx++) {
      NESOASSERT(this->ptr_bit_tmp[cx] == static_cast<unsigned char>(255),
                 "DEBUG PADDING END TOUCHED");
    }

#endif
  }
};

typedef std::shared_ptr<SYCLTarget> SYCLTargetSharedPtr;

/**
 * Container around USM device allocated memory that can be resized.
 */
template <typename T> class BufferDevice {
private:
public:
  /// Disable (implicit) copies.
  BufferDevice(const BufferDevice &st) = delete;
  /// Disable (implicit) copies.
  BufferDevice &operator=(BufferDevice const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// SYCL USM device pointer, only accessible on device.
  T *ptr;
  /// Number of elements allocated.
  size_t size;

  /**
   * Create a new BufferDevice of a given number of elements.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param size Number of elements.
   */
  BufferDevice(SYCLTargetSharedPtr &sycl_target, size_t size)
      : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)this->sycl_target->malloc_device(size * sizeof(T));
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
    if (this->size > 0) {
      this->sycl_target->queue.memcpy(this->ptr, vec.data(), this->size_bytes())
          .wait();

      auto k_ptr = this->ptr;

      sycl::buffer<T, 1> b_vec(vec.data(), vec.size());
      sycl_target->queue
          .submit([&](sycl::handler &cgh) {
            auto a_vec =
                b_vec.template get_access<sycl::access::mode::read>(cgh);
            cgh.parallel_for<>(
                sycl::range<1>(this->size),
                [=](sycl::id<1> idx) { k_ptr[idx] = a_vec[idx]; });
          })
          .wait_and_throw();
    }
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
      this->sycl_target->free(this->ptr);
      this->ptr = (T *)this->sycl_target->malloc_device(size * sizeof(T));
      this->size = size;
    }
    return this->size;
  }
  ~BufferDevice() {
    if (this->ptr != NULL) {
      this->sycl_target->free(this->ptr);
    }
  }
};

/**
 * Container around USM shared allocated memory that can be resized.
 */
template <typename T> class BufferShared {
private:
public:
  /// Disable (implicit) copies.
  BufferShared(const BufferShared &st) = delete;
  /// Disable (implicit) copies.
  BufferShared &operator=(BufferShared const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// SYCL USM shared pointer, accessible on host and device.
  T *ptr;
  /// Number of elements allocated.
  size_t size;

  /**
   * Create a new DeviceShared of a given number of elements.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param size Number of elements.
   */
  BufferShared(SYCLTargetSharedPtr sycl_target, size_t size)
      : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl::malloc_shared(size * sizeof(T), sycl_target->queue);
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
      sycl::free(this->ptr, this->sycl_target->queue);
      this->ptr =
          (T *)sycl::malloc_shared(size * sizeof(T), sycl_target->queue);
      this->size = size;
    }
    return this->size;
  }

  ~BufferShared() {
    if (this->ptr != NULL) {
      sycl::free(this->ptr, sycl_target->queue);
    }
  }
};

/**
 * Container around USM host allocated memory that can be resized.
 */
template <typename T> class BufferHost {
private:
public:
  /// Disable (implicit) copies.
  BufferHost(const BufferHost &st) = delete;
  /// Disable (implicit) copies.
  BufferHost &operator=(BufferHost const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// SYCL USM shared pointer, accessible on host and device.
  T *ptr;
  /// Number of elements allocated.
  size_t size;

  /**
   * Create a new BufferHost of a given number of elements.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param size Number of elements.
   */
  BufferHost(SYCLTargetSharedPtr sycl_target, size_t size)
      : sycl_target(sycl_target) {
    this->size = size;
    this->ptr = (T *)sycl_target->malloc_host(size * sizeof(T));
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
    if (this->size > 0) {
      this->sycl_target->queue.memcpy(this->ptr, vec.data(), this->size_bytes())
          .wait();
    }
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
      sycl_target->free(this->ptr);
      this->ptr = (T *)sycl_target->malloc_host(size * sizeof(T));
      this->size = size;
    }
    return this->size;
  }
  ~BufferHost() {
    if (this->ptr != NULL) {
      sycl_target->free(this->ptr);
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
