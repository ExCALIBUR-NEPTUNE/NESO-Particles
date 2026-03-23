#ifndef _NESO_PARTICLES_COMPUTE_TARGET
#define _NESO_PARTICLES_COMPUTE_TARGET

#include <array>
#include <cstdlib>
#include <map>
#include <mpi.h>
#include <optional>
#include <stack>
#include <string>
#include <vector>

#include "communication.hpp"
#include "containers/resource_stack_map.hpp"
#include "device_functions.hpp"
#include "device_limits.hpp"
#include "parameters.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  Determine a local MPI rank based on environment variables and shared memory
 *  splitting.
 *
 *  @param comm  MPI_Comm to try and deduce local rank from.
 *  @param default_rank (optional)  MPI rank to use if one cannot be determined
 *  from environment variables or SHM intra comm.
 */
int get_local_mpi_rank(MPI_Comm comm, int default_rank = -1);

/**
 * Container for SYCL devices and queues such that they can be easily passed
 * around.
 */
class SYCLTarget {
private:
  std::map<unsigned char *, std::size_t> ptr_map;

#ifdef DEBUG_OOB_CHECK
  std::array<unsigned char, DEBUG_OOB_WIDTH> ptr_bit_mask;
  std::array<unsigned char, DEBUG_OOB_WIDTH> ptr_bit_tmp;
#endif

  int num_devices{0};
  int local_rank{-1};
  std::string auto_profiling_prefix{""};

  void print_info_inner();
  std::size_t get_local_size();

public:
  /// SYCL device in use.
  sycl::device device;
  /// The local index of the device in the platform.
  int device_index{-1};
  /// Main SYCL queue to use.
  sycl::queue queue;
  /// Parent MPI communicator to use.
  MPI_Comm comm;
  /// CommPair to pass around inter and intra communicators for shared memory
  /// regions.
  CommPair comm_pair;
  /// ProfileMap to log profiling data related to this SYCLTarget.
  ProfileMap profile_map;
  /// Parameters for generic properties, e.g. loop local sizes.
  std::shared_ptr<Parameters> parameters;
  /// Interface to device limits validation
  DeviceLimits device_limits;
  /// A ResourceStackMap that is available to downstream types.
  std::shared_ptr<ResourceStackMap> resource_stack_map;

  /// Disable (implicit) copies.
  SYCLTarget(const SYCLTarget &st) = delete;
  /// Disable (implicit) copies.
  SYCLTarget &operator=(SYCLTarget const &a) = delete;

// Add a define to use SYCL 1.2 selectors if 2020 ones are not supported
#if (SYCL_LANGUAGE_VERSION == 202001) && defined(__INTEL_LLVM_COMPILER)
#if (__INTEL_LLVM_COMPILER < 20230000)
#define NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
#endif
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
  SYCLTarget(const int gpu_device, MPI_Comm comm, int local_rank = -1);

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
  void print_device_info();

  /**
   * Print information to stdout from all ranks. Collective on the communicator.
   */
  void print_world_device_info();

  /**
   * Free the SYCLTarget and underlying CommPair.
   */
  void free();

  /**
   * Allocate memory on device using sycl::malloc_device.
   *
   * @param size_bytes Number of bytes to allocate.
   * @param align_bytes Optional alignment, default 0 calls malloc_device
   * instead of aligned_alloc_device.
   */
  void *malloc_device(const std::size_t size_bytes,
                      const std::size_t align_bytes = 0);

  /**
   * Allocate memory in USM shared memory using sycl::malloc_shared.
   *
   * @param size_bytes Number of bytes to allocate.
   * @param align_bytes Optional alignment, default 0 calls malloc_shared
   * instead of aligned_alloc_shared.
   */
  void *malloc_shared(const std::size_t size_bytes,
                      const std::size_t align_bytes = 0);

  /**
   * Allocate memory on the host using sycl::malloc_host.
   *
   * @param size_bytes Number of bytes to allocate.
   * @param align_bytes Optional alignment, default 0 calls malloc_host instead
   * of aligned_alloc_host.
   */
  void *malloc_host(const std::size_t size_bytes,
                    const std::size_t align_bytes = 0);

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

  void check_ptrs();

  void check_ptr([[maybe_unused]] unsigned char *ptr_user,
                 [[maybe_unused]] const std::size_t size_bytes);

  /**
   *  Get a number of local work items that should not exceed the maximum
   *  available local memory on the device.
   *
   *  @param num_bytes Number of bytes requested per work item.
   *  @param default_num Default number of work items.
   *  @returns Number of work items.
   */
  std::size_t get_num_local_work_items(const std::size_t num_bytes,
                                       const std::size_t default_num);

  /**
   *  Get a number of local work items that should not exceed the maximum
   *  available local memory on the device.
   *
   *  @param num_bytes_offset Number of bytes requested for the work group.
   *  @param num_bytes Number of bytes requested per work item.
   *  @param default_num Default number of work items.
   *  @returns Number of work items.
   */
  std::size_t get_num_local_work_items(const std::size_t num_bytes_offset,
                                       const std::size_t num_bytes,
                                       const std::size_t default_num);
};

typedef std::shared_ptr<SYCLTarget> SYCLTargetSharedPtr;

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
  const std::size_t offset;
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
NDRangePeel1D get_nd_range_peel_1d(const std::size_t size,
                                   const std::size_t local_size);

} // namespace NESO::Particles

#endif
