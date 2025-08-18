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

  void print_info_inner();

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

/**
 * Compute the exclusive scan of an array using the SYCL group built-ins.
 *
 * @param[in] sycl_target Compute device to use.
 * @param[in] N Number of elements.
 * @param[in] d_src Device pointer to source values.
 * @param[in, out] d_dst Device pointer to destination values.
 * @returns Event to wait on for completion.
 */
template <typename T>
[[nodiscard]] inline sycl::event
joint_exclusive_scan(SYCLTargetSharedPtr sycl_target, std::size_t N, T *d_src,
                     T *d_dst) {
  if (N == 0) {
    return sycl::event{};
  }

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  NESOASSERT(local_size >= 1, "Bad group size for exclusive_scan.");
  return sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(local_size),
                                       sycl::range<1>(local_size)),
                     [=](sycl::nd_item<1> it) {
                       T *first = d_src;
                       T *last = first + N;
                       sycl::joint_exclusive_scan(it.get_group(), first, last,
                                                  d_dst, sycl::plus<T>());
                     });
  });
}

extern template sycl::event
joint_exclusive_scan(SYCLTargetSharedPtr sycl_target, std::size_t N, int *d_src,
                     int *d_dst);
extern template sycl::event
joint_exclusive_scan(SYCLTargetSharedPtr sycl_target, std::size_t N, INT *d_src,
                     INT *d_dst);

/**
 * Compute the exclusive scan of a n arrays using the SYCL group built-ins.
 *
 * @param[in] sycl_target Compute device to use.
 * @param[in] N Number of arrays.
 * @param[in] d_array_sizes Number of elements in each sub array.
 * @param[in] d_array_offsets The starting index of each sub array.
 * @param[in] d_src Device poitner to source values.
 * @param[in, out] d_dst Device pointer to destination values  (same size as
 * d_src).
 * @returns Event to wait on for completion.
 */
template <typename U, typename T>
[[nodiscard]] inline sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       const U *RESTRICT const d_array_sizes,
                       const U *RESTRICT const d_array_offsets, T *d_src,
                       T *d_dst)

{
  if (N == 0) {
    return sycl::event{};
  }

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  auto iteration_set =
      sycl_target->device_limits.validate_nd_range(sycl::nd_range<2>(
          sycl::range<2>(N, local_size), sycl::range<2>(1, local_size)));

  NESOASSERT(local_size >= 1, "Bad local size for exclusive_scan.");
  return sycl_target->queue.parallel_for(
      iteration_set, [=](sycl::nd_item<2> it) {
        const std::size_t array_index = it.get_global_id(0);
        const auto num_elements = d_array_sizes[array_index];
        if (num_elements > 0) {
          const auto start_index = d_array_offsets[array_index];
          T *first = d_src + start_index;
          T *last = first + num_elements;
          sycl::joint_exclusive_scan(it.get_group(), first, last,
                                     d_dst + start_index, sycl::plus<T>());
        }
      });
}

extern template sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       const int *RESTRICT const d_array_sizes,
                       const int *RESTRICT const d_array_offsets, int *d_src,
                       int *d_dst);

extern template sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       const INT *RESTRICT const d_array_sizes,
                       const INT *RESTRICT const d_array_offsets, INT *d_src,
                       INT *d_dst);

/**
 * Compute the exclusive scan of a n arrays using the SYCL group built-ins. Also
 * computes the total for each of the n arrays.
 *
 * @param[in] sycl_target Compute device to use.
 * @param[in] N Number of arrays.
 * @param[in] d_array_sizes Number of elements in each sub array.
 * @param[in] d_array_offsets The starting index of each sub array.
 * @param[in] d_src Device pointer to source values.
 * @param[in, out] d_dst Device pointer to destination values (same size as
 * d_src).
 * @param[in, out] d_dst_sum Device pointer to destination summation_values
 * (size n).
 * @returns Event to wait on for completion.
 */
template <typename U, typename T>
[[nodiscard]] inline sycl::event
joint_exclusive_scan_n_sum(SYCLTargetSharedPtr sycl_target, std::size_t N,
                           const U *RESTRICT const d_array_sizes,
                           const U *RESTRICT const d_array_offsets, T *d_src,
                           T *d_dst, T *d_dst_sum) {
  if (N == 0) {
    return sycl::event{};
  }

  sycl::event event_es = joint_exclusive_scan_n(sycl_target, N, d_array_sizes,
                                                d_array_offsets, d_src, d_dst);

  // This loop is dependent on the exclusive scan call above.
  sycl::event event_totals = sycl_target->queue.parallel_for(
      sycl::range<1>(N), event_es, [=](auto ix) {
        if (d_array_sizes[ix] == 0) {
          d_dst_sum[ix] = 0;
        } else {
          const auto last_value =
              d_src[d_array_offsets[ix] + d_array_sizes[ix] - 1];
          const auto last_value_ex =
              d_dst[d_array_offsets[ix] + d_array_sizes[ix] - 1];
          d_dst_sum[ix] = last_value + last_value_ex;
        }
      });

  return event_totals;
}

extern template sycl::event
joint_exclusive_scan_n_sum(SYCLTargetSharedPtr sycl_target, std::size_t N,
                           const int *RESTRICT const d_array_sizes,
                           const int *RESTRICT const d_array_offsets,
                           int *d_src, int *d_dst, int *d_dst_sum);

extern template sycl::event
joint_exclusive_scan_n_sum(SYCLTargetSharedPtr sycl_target, std::size_t N,
                           const INT *RESTRICT const d_array_sizes,
                           const INT *RESTRICT const d_array_offsets,
                           INT *d_src, INT *d_dst, INT *d_dst_sum);

/**
 * Compute the transpose of a matrix stored in row major format. Assumes that
 * the matrix might be very non-square.
 *
 * @param[in] sycl_target Compute device to use.
 * @param[in] num_rows Number of rows.
 * @param[in] num_cols Number of columns.
 * @param[in] d_src Device pointer to input matrix.
 * @param[in, out] d_dst Device pointer to output matrix.
 * @returns Event to wait on for completion.
 */
template <typename T>
[[nodiscard]] inline sycl::event
matrix_transpose(SYCLTargetSharedPtr sycl_target, const std::size_t num_rows,
                 const std::size_t num_cols, const T *RESTRICT const d_src,
                 T *RESTRICT d_dst) {
  if ((num_rows <= 1) || (num_cols <= 1)) {
    return sycl::event();
  }
  const std::size_t num_bytes_per_item = sizeof(T);
  std::size_t local_size =
      sycl_target->parameters->get<SizeTParameter>("LOOP_LOCAL_SIZE")->value;
  local_size =
      sycl_target->get_num_local_work_items(num_bytes_per_item, local_size);
  local_size = std::sqrt(local_size);

  const std::size_t local_size_row =
      get_prev_power_of_two(std::min(num_rows, local_size));
  const std::size_t local_size_col =
      get_prev_power_of_two(std::min(num_cols, local_size));

  sycl::range<2> range_local(local_size_row, local_size_col);
  sycl::range<2> range_global(get_next_multiple(num_rows, local_size_row),
                              get_next_multiple(num_cols, local_size_col));

  return sycl_target->queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<T, 1> local_memory(
        sycl::range<1>(local_size_row * local_size_col), cgh);
    cgh.parallel_for(
        sycl_target->device_limits.validate_nd_range(
            sycl::nd_range<2>(range_global, range_local)),
        [=](sycl::nd_item<2> item) {
          const std::size_t read_rowx = item.get_global_id(0);
          const std::size_t read_colx = item.get_global_id(1);

          const std::size_t local_read_rowx = item.get_local_id(0);
          const std::size_t local_read_colx = item.get_local_id(1);

          const bool read_item_valid =
              (read_rowx < num_rows) && (read_colx < num_cols);

          T value_read = 0.0;
          if (read_item_valid) {
            value_read = d_src[read_rowx * num_cols + read_colx];
          }

          // This might have shared memory bank conflicts on GPUs
          local_memory[local_read_rowx * local_size_col + local_read_colx] =
              value_read;
          item.barrier(sycl::access::fence_space::local_space);

          const std::size_t local_write_rowx =
              item.get_local_linear_id() / local_size_row;
          const std::size_t local_write_colx =
              item.get_local_linear_id() % local_size_row;

          // This might have shared memory bank conflicts on GPUs
          const T value_write = local_memory[local_write_colx * local_size_col +
                                             local_write_rowx];

          // compute write index
          const std::size_t group_row = item.get_group(0);
          const std::size_t group_col = item.get_group(1);
          const std::size_t write_rowx =
              group_col * local_size_col + local_write_rowx;
          const std::size_t write_colx =
              group_row * local_size_row + local_write_colx;

          const bool write_item_valid =
              (write_rowx < num_cols) && (write_colx < num_rows);

          if (write_item_valid) {
            d_dst[write_rowx * num_rows + write_colx] = value_write;
          }
        });
  });
}

extern template sycl::event matrix_transpose(SYCLTargetSharedPtr sycl_target,
                                             const std::size_t num_rows,
                                             const std::size_t num_cols,
                                             const REAL *RESTRICT const d_src,
                                             REAL *RESTRICT d_dst);

/**
 * @param ptr Pointer to align to alignment.
 * @param alignment Power of two to align to.
 * @returns Aligned pointer.
 */
template <typename T>
constexpr inline T *cast_align_pointer(void *ptr, const std::size_t alignment) {
  std::size_t ptr_int = reinterpret_cast<std::size_t>(ptr);
  ptr_int = (ptr_int + (alignment - 1)) & -alignment;
  return reinterpret_cast<T *>(ptr_int);
}

} // namespace NESO::Particles

#endif
