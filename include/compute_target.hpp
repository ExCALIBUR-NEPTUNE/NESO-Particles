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

// Add a define to use SYCL 1.2 selectors if 2020 ones are not supported
#if (SYCL_LANGUAGE_VERSION == 202001) && defined(__INTEL_LLVM_COMPILER)
#define NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
#endif

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
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
        this->device = sycl::device{sycl::gpu_selector()};
#else
        this->device = sycl::device{sycl::gpu_selector_v};
#endif
      } catch (sycl::exception const &e) {
        std::cout << "Cannot select a GPU\n" << e.what() << "\n";
        std::cout << "Using a CPU device\n";
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
        this->device = sycl::device{sycl::cpu_selector()};
#else
        this->device = sycl::device{sycl::cpu_selector_v};
#endif
      }
    } else if (gpu_device < 0) {
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
      this->device = sycl::device{sycl::cpu_selector()};
#else
      this->device = sycl::device{sycl::cpu_selector_v};
#endif
    } else {

      // Get the default device and platform as they are most likely to be the
      // desired device based on SYCL implementation/runtime/environment
      // variables.
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
      sycl::device default_device{sycl::default_selector()};
#else
      sycl::device default_device{sycl::default_selector_v};
#endif
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
      : sycl_target(sycl_target), size(size), ptr(nullptr) {}

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
   */
  inline int realloc_no_copy(const std::size_t size) {
    if (size > this->size) {
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

  ~BufferDevice() { this->generic_free(); }
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
    sycl::free(this->ptr, this->sycl_target->queue);
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
 * Get an 1D nd_range for a given iteration set global size and local size.
 *
 * @param size Global iteration set size.
 * @param local_size Local iteration set size (work group size).
 * @returns nd_range large enough to cover global iteration set. May be larger
 * than size.
 */
inline sycl::nd_range<1> get_nd_range_1d(const std::size_t size,
                                         const std::size_t local_size) {
  const auto div_mod = std::div(static_cast<long long>(size),
                                static_cast<long long>(local_size));
  const std::size_t outer_size =
      static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1));
  return sycl::nd_range(sycl::range<1>(outer_size * local_size),
                        sycl::range<1>(local_size));
}

/**
 *  Main loop with peel loop for 1D nd_range. The peel loop should be masked to
 *  remove the workitems past the loop extents.
 */
struct NDRangePeel1D {
  /// The main iteration set which does not need masking.
  sycl::nd_range<1> loop_main;
  /// Bool to indicate if there is a peel loop.
  const bool peel_exists;
  /// Offset to apply to peel loop indices.
  const size_t offset;
  /// Peel loop description. Indicies from the loop should have offset added to
  /// compute the global index. This loop should be masked by a conditional for
  /// the original loop bounds.
  sycl::nd_range<1> loop_peel;
};

/**
 * Create two nd_ranges that cover a 1D iteration set. The first is a main
 * iteration set that does not need a mask. The second is an iteration set that
 * requires 1) the offset adding to the iteration index and 2) a conditional to
 * test the resulting index is less than the loop bound.
 *
 * @param size Global iteration set size.
 * @param local_size Local iteration set size (SYCL workgroup size).
 * @returns NDRangePeel1D instance describing loop iteration sets.
 */
inline NDRangePeel1D get_nd_range_peel_1d(const std::size_t size,
                                          const std::size_t local_size) {
  const auto div_mod = std::div(static_cast<long long>(size),
                                static_cast<long long>(local_size));

  const std::size_t outer_size =
      static_cast<std::size_t>(div_mod.quot) * local_size;
  const bool peel_exists = !(div_mod.rem == 0);
  const std::size_t outer_size_peel = (peel_exists ? 1 : 0) * local_size;
  const size_t offset = outer_size;

  return NDRangePeel1D{
      sycl::nd_range<1>(sycl::range<1>(outer_size), sycl::range<1>(local_size)),
      peel_exists, offset,
      sycl::nd_range<1>(sycl::range<1>(outer_size_peel),
                        sycl::range<1>(local_size))};
}

} // namespace NESO::Particles

#endif
