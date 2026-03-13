#ifndef _NESO_PARTICLES_ALGORITHMS_COMMON_HPP_
#define _NESO_PARTICLES_ALGORITHMS_COMMON_HPP_

#include "../compute_target.hpp"

namespace NESO::Particles {

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
                       sycl::group_barrier(it.get_group());
                       if (it.get_global_linear_id() == 0) {
                         d_dst[0] = 0;
                       }
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
 * Compute the number of blocks required for a larger exclusive scan.
 *
 * @param sycl_target SYCLTarget to use for exclusive scan.
 * @param N Number of elements.
 * @returns Required number of blocks.
 */
std::size_t
get_joint_exclusive_scan_aux_num_blocks(SYCLTargetSharedPtr sycl_target,
                                        const std::size_t N);

/**
 * Compute the size of the auxillary buffer required for a larger exclusive
 * scan.
 *
 * @param sycl_target SYCLTarget to use for exclusive scan.
 * @param N Number of elements.
 * @returns Required size of auxillary array.
 */
std::size_t
get_joint_exclusive_scan_aux_array_size(SYCLTargetSharedPtr sycl_target,
                                        const std::size_t N);

/**
 * Compute the exclusive scan of an array using the SYCL group built-ins and an
 * auxillary array.
 *
 * @param[in] sycl_target Compute device to use.
 * @param[in] N Number of elements.
 * @param[in, out] d_aux Device pointer to auxillary buffer. Will be modified.
 * Must not be freed until the returned event is complete.
 * @param[in] d_src Device pointer to source values.
 * @param[in, out] d_dst Device pointer to destination values.
 * @returns Event to wait on for completion.
 */
template <typename T>
[[nodiscard]] inline sycl::event
joint_exclusive_scan(SYCLTargetSharedPtr sycl_target, std::size_t N,
                     T *RESTRICT d_aux, T *RESTRICT d_src, T *RESTRICT d_dst) {
  if (N == 0) {
    return sycl::event{};
  }

  const std::size_t num_blocks =
      get_joint_exclusive_scan_aux_num_blocks(sycl_target, N);

  if (num_blocks < 2) {
    return joint_exclusive_scan(sycl_target, N, d_src, d_dst);
  }

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  auto iteration_set0 = sycl_target->device_limits.validate_nd_range(
      sycl::nd_range<2>(sycl::range<2>(num_blocks, local_size),
                        sycl::range<2>(1, local_size)));

  auto e0 =
      sycl_target->queue.parallel_for(iteration_set0, [=](sycl::nd_item<2> it) {
        const std::size_t block_index = it.get_global_id(0);

        std::size_t start_index = 0;
        std::size_t end_index = 0;
        get_decomp_1d(num_blocks, N, block_index, &start_index, &end_index);

        T *first = d_src + start_index;
        T *last = d_src + end_index;
        sycl::joint_exclusive_scan(it.get_group(), first, last,
                                   d_dst + start_index, sycl::plus<T>());
        sycl::group_barrier(it.get_group());
        if (it.get_local_linear_id() == 0) {
          d_dst[start_index] = 0;
        }
      });

  auto e1 = sycl_target->queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
      e0, [=](sycl::nd_item<1> it) {
        {
          const std::size_t local_index = it.get_local_id(0);
          for (std::size_t ix = local_index; ix < num_blocks;
               ix += local_size) {

            std::size_t start_index = 0;
            std::size_t end_index = 0;
            get_decomp_1d(num_blocks, N, ix, &start_index, &end_index);

            std::size_t last_index = end_index - 1;
            const T value = d_dst[last_index] + d_src[last_index];
            d_aux[ix] = value;
          }
        }
        sycl::group_barrier(it.get_group());
        {
          T *first = d_aux;
          T *last = first + num_blocks;
          sycl::joint_exclusive_scan(it.get_group(), first, last, last,
                                     sycl::plus<T>());
          sycl::group_barrier(it.get_group());
          if (it.get_global_linear_id() == 0) {
            d_aux[num_blocks] = 0;
          }
        }
      });

  auto iteration_set1 = sycl_target->device_limits.validate_nd_range(
      sycl::nd_range<2>(sycl::range<2>(num_blocks - 1, local_size),
                        sycl::range<2>(1, local_size)));
  auto e2 = sycl_target->queue.parallel_for(
      iteration_set1, e1, [=](sycl::nd_item<2> it) {
        const std::size_t block_index = it.get_global_id(0) + 1;
        const std::size_t local_index = it.get_local_id(1);

        std::size_t start_index = 0;
        std::size_t end_index = 0;
        get_decomp_1d(num_blocks, N, block_index, &start_index, &end_index);

        const T shift = d_aux[num_blocks + block_index];

        for (std::size_t ix = (local_index + start_index); ix < end_index;
             ix += local_size) {
          d_dst[ix] += shift;
        }
      });

  return e2;
}

/**
 * Compute the exclusive scan of a n arrays using the SYCL group built-ins.
 *
 * @param[in] sycl_target Compute device to use.
 * @param[in] N Number of arrays.
 * @param[in] d_array_sizes Number of elements in each sub array.
 * @param[in] d_array_offsets The starting index of each sub array.
 * @param[in] d_src Device pointer to source values.
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
          sycl::group_barrier(it.get_group());
          if (it.get_local_linear_id() == 0) {
            d_dst[start_index] = 0;
          }
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
 * Compute the exclusive scan of a n arrays using the SYCL group built-ins.
 *
 * @param[in] sycl_target Compute device to use.
 * @param[in] N Number of arrays.
 * @param[in] M Number of elements in each sub array.
 * @param[in] d_src Device poitner to source values.
 * @param[in, out] d_dst Device pointer to destination values  (same size as
 * d_src).
 * @returns Event to wait on for completion.
 */
template <typename T>
[[nodiscard]] inline sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       std::size_t M, T *d_src, T *d_dst) {
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
        if (M > 0) {
          const auto start_index = M * array_index;
          T *first = d_src + start_index;
          T *last = first + M;
          sycl::joint_exclusive_scan(it.get_group(), first, last,
                                     d_dst + start_index, sycl::plus<T>());
          sycl::group_barrier(it.get_group());
          if (it.get_local_linear_id() == 0) {
            d_dst[start_index] = 0;
          }
        }
      });
}

/**
 * Compute the inclusive scan of a n arrays using the SYCL group built-ins.
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
joint_inclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
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
          sycl::joint_inclusive_scan(it.get_group(), first, last,
                                     d_dst + start_index, sycl::plus<T>());
          sycl::group_barrier(it.get_group());
        }
      });
}

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

/**
 * Compute reduce the entries of an array.
 *
 * @param[in] sycl_target Compute device.
 * @param[in] n Number of items in buffer.
 * @param[in] ptr Pointer to start of input buffer.
 * @param[in] binary_op Binary operation to use for reduction.
 * @param[in, out] result_ptr Output location for reduction.
 * @returns Event to wait on for completion.
 */
template <typename T, typename BINARY_OP>
[[nodiscard]] inline sycl::event
reduce_values(SYCLTargetSharedPtr sycl_target, const std::size_t n,
              T *RESTRICT ptr, BINARY_OP binary_op, T *RESTRICT result_ptr) {
  std::size_t local_size =
      sycl_target->parameters->get<SizeTParameter>("LOOP_LOCAL_SIZE")->value;

  return sycl_target->queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(local_size), sycl::range<1>(local_size)),
      [=](sycl::nd_item<1> idx) {
        const T v =
            Kernel::joint_reduce(idx.get_group(), ptr, ptr + n, binary_op);
        if (idx.get_global_linear_id() == 0) {
          result_ptr[0] = v;
        }
      });
}

} // namespace NESO::Particles

#endif
