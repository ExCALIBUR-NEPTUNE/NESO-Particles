#include <neso_particles/algorithms/common.hpp>

namespace NESO::Particles {

template sycl::event joint_exclusive_scan(SYCLTargetSharedPtr sycl_target,
                                          std::size_t N, int *d_src,
                                          int *d_dst);
template sycl::event joint_exclusive_scan(SYCLTargetSharedPtr sycl_target,
                                          std::size_t N, INT *d_src,
                                          INT *d_dst);

std::size_t
get_joint_exclusive_scan_aux_num_blocks(SYCLTargetSharedPtr sycl_target,
                                        const std::size_t N) {

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  const std::size_t max_compute_units =
      sycl_target->parameters->template get<SizeTParameter>("MAX_COMPUTE_UNITS")
          ->value;

  return std::min(div_round_up(N, local_size), max_compute_units);
}

std::size_t
get_joint_exclusive_scan_aux_array_size(SYCLTargetSharedPtr sycl_target,
                                        const std::size_t N) {
  return 2 * get_joint_exclusive_scan_aux_num_blocks(sycl_target, N);
}

template sycl::event
joint_exclusive_scan_n_sum(SYCLTargetSharedPtr sycl_target, std::size_t N,
                           const int *RESTRICT const d_array_sizes,
                           const int *RESTRICT const d_array_offsets,
                           int *d_src, int *d_dst, int *d_dst_sum);

template sycl::event
joint_exclusive_scan_n_sum(SYCLTargetSharedPtr sycl_target, std::size_t N,
                           const INT *RESTRICT const d_array_sizes,
                           const INT *RESTRICT const d_array_offsets,
                           INT *d_src, INT *d_dst, INT *d_dst_sum);

template sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       const int *RESTRICT const d_array_sizes,
                       const int *RESTRICT const d_array_offsets, int *d_src,
                       int *d_dst);

template sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       const INT *RESTRICT const d_array_sizes,
                       const INT *RESTRICT const d_array_offsets, INT *d_src,
                       INT *d_dst);

template sycl::event matrix_transpose(SYCLTargetSharedPtr sycl_target,
                                      const std::size_t num_rows,
                                      const std::size_t num_cols,
                                      const REAL *RESTRICT const d_src,
                                      REAL *RESTRICT d_dst);

} // namespace NESO::Particles
