#ifndef _NESO_PARTICLES_DEVICE_FUNCTIONS_HPP_
#define _NESO_PARTICLES_DEVICE_FUNCTIONS_HPP_

#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

namespace Kernel {

template <typename T, typename U> inline auto min(const T &x, const U &y) {
  return KERNEL_MIN(x, y);
}
inline auto min(const REAL &x, const REAL &y) { return sycl::fmin(x, y); }
template <typename T, typename U> inline auto max(const T &x, const U &y) {
  return KERNEL_MIN(x, y);
}
inline auto max(const REAL &x, const REAL &y) { return sycl::fmax(x, y); }
template <typename T> inline auto abs(const T &x) { return KERNEL_ABS(x); }
inline auto abs(const REAL &x) { return sycl::fabs(x); }
inline auto sqrt(const REAL &x) { return sycl::sqrt(x); }
inline auto rsqrt(const REAL &x) { return sycl::rsqrt(x); }
inline auto exp(const REAL &x) { return sycl::exp(x); }
inline auto fmod(const REAL &x, const REAL &y) { return sycl::fmod(x, y); }
inline auto fma(const REAL &x, const REAL &y, const REAL &z) {
  return sycl::fma(x, y, z);
}

} // namespace Kernel

/**
 * Compute the intersection point parameter (lambda0, lambda1) for the lines
 * [(xa, ya), (xb, yb)] and [(x0, y0), (x1, y1)].
 * Assuming that x1 - x0 != 0 and that the lines are not parallel.
 *
 * param[in] xa Input coordinate for point a.
 * param[in] ya Input coordinate for point a.
 * param[in] xb Input coordinate for point b.
 * param[in] yb Input coordinate for point b.
 * param[in] x0 Input coordinate for point 0.
 * param[in] y0 Input coordinate for point 0.
 * param[in] x1 Input coordinate for point 1.
 * param[in] y1 Input coordinate for point 1.
 * param[in, out] l0 Output intersection lambda0 if it exists.
 * param[in, out] l1 Output intersection lambda1 if it exists.
 */
inline void line_segment_intersection_2d_lambda(const REAL &xa, const REAL &ya,
                                                const REAL &xb, const REAL &yb,
                                                const REAL &x0, const REAL &y0,
                                                const REAL &x1, const REAL &y1,
                                                REAL &l0, REAL &l1) {

  /**
   * Solve
   * 0 = xa - x0 + l0 * (xb - xa) - l1 * (x1 - x0) and
   * 0 = ya - y0 + l0 * (yb - ya) - l1 * (y1 - y0)
   * for l0.
   */
  const REAL inverse_x1mx0 = 1.0 / (x1 - x0);
  l0 = (-ya + y0 + (xa - x0) * (y1 - y0) * inverse_x1mx0) /
       (yb - ya - (xb - xa) * (y1 - y0) * inverse_x1mx0);
  l1 = (xa - x0 + l0 * (xb - xa)) * inverse_x1mx0;
}

/**
 * Compute the intersection point of two line segments
 * [(xa, ya), (xb, yb)] and [(x0, y0), (x1, y1)].
 *
 * param[in] xa Input coordinate for point a.
 * param[in] ya Input coordinate for point a.
 * param[in] xb Input coordinate for point b.
 * param[in] yb Input coordinate for point b.
 * param[in] x0 Input coordinate for point 0.
 * param[in] y0 Input coordinate for point 0.
 * param[in] x1 Input coordinate for point 1.
 * param[in] y1 Input coordinate for point 1.
 * param[in, out] xi Output intersection point if it exists.
 * param[in, out] yi Output intersection point if it exists.
 * param[in, out] l0_out Proportion of the distance between a and b the
 * intersection point exists at.
 * param[in] tol Tolerance for intersection, e.g. how closely do the lines pass
 * at the ends default 0.0.
 * @returns True if the line segments intersect otherwise false.
 */
inline bool line_segment_intersection_2d(const REAL &xa, const REAL &ya,
                                         const REAL &xb, const REAL &yb,
                                         const REAL &x0, const REAL &y0,
                                         const REAL &x1, const REAL &y1,
                                         REAL &xi, REAL &yi, REAL &l0_out,
                                         const REAL tol = 0.0) {

  const REAL vabx = xb - xa;
  const REAL vaby = yb - ya;
  const REAL v01x = x1 - x0;
  const REAL v01y = y1 - y0;
  // a1 b2 - a2 b1
  const REAL cross_product_z = vabx * v01y - vaby * v01x;
  // Check the lines are not parallel.
  if (cross_product_z != 0.0) {
    REAL l0, l1;
    if (KERNEL_ABS(x1 - x0) > KERNEL_ABS(xb - xa)) {
      line_segment_intersection_2d_lambda(xa, ya, xb, yb, x0, y0, x1, y1, l0,
                                          l1);
      xi = xa + l0 * (xb - xa);
      yi = ya + l0 * (yb - ya);
      l0_out = l0;
    } else {
      line_segment_intersection_2d_lambda(x0, y0, x1, y1, xa, ya, xb, yb, l0,
                                          l1);
      xi = xa + l1 * (xb - xa);
      yi = ya + l1 * (yb - ya);
      l0_out = l1;
    }

    // This segment where the ends are padded with +- tol.
    const bool in_bounds_ab =
        ((0.0 - tol) <= l0_out) && (l0_out <= (1.0 + tol));

    // This segment is the one where we want strict bounds.
    const bool x_cond = (x0 < x1);
    const REAL x_low = x_cond ? x0 : x1;
    const REAL x_max = x_cond ? x1 : x0;
    const bool y_cond = (y0 < y1);
    const REAL y_low = y_cond ? y0 : y1;
    const REAL y_max = y_cond ? y1 : y0;

    const bool in_bounds_01 =
        (x_low <= xi) && (xi <= x_max) && (y_low <= yi) && (yi <= y_max);

    return in_bounds_ab && in_bounds_01;
  } else {
    return false;
  }
}

/**
 * Naively invert a matrix. The error bars on this call may be quite large.
 * This function uses row-major format.
 *
 * @param[in] M Matrix to invert.
 * @param[in, out] L Output space for M^-1.
 */
template <std::size_t N>
inline void naive_matrix_inverse([[maybe_unused]] const REAL *RESTRICT M,
                                 [[maybe_unused]] REAL *RESTRICT L) {
  static_assert(N == -1, "Not implemented, see specialisations.");
}

template <>
inline void naive_matrix_inverse<3>(const REAL *RESTRICT M, REAL *RESTRICT L) {
  const REAL inverse_factor =
      1.0 / (M[0] * M[4] * M[8] - M[0] * M[5] * M[7] - M[1] * M[3] * M[8] +
             M[1] * M[5] * M[6] + M[2] * M[3] * M[7] - M[2] * M[4] * M[6]);
  L[0] = (M[4] * M[8] - M[5] * M[7]) * inverse_factor;
  L[1] = (-M[1] * M[8] + M[2] * M[7]) * inverse_factor;
  L[2] = (M[1] * M[5] - M[2] * M[4]) * inverse_factor;
  L[3] = (-M[3] * M[8] + M[5] * M[6]) * inverse_factor;
  L[4] = (M[0] * M[8] - M[2] * M[6]) * inverse_factor;
  L[5] = (-M[0] * M[5] + M[2] * M[3]) * inverse_factor;
  L[6] = (M[3] * M[7] - M[4] * M[6]) * inverse_factor;
  L[7] = (-M[0] * M[7] + M[1] * M[6]) * inverse_factor;
  L[8] = (M[0] * M[4] - M[1] * M[3]) * inverse_factor;
}

template <>
inline void naive_matrix_inverse<4>(const REAL *RESTRICT M, REAL *RESTRICT L) {
  const REAL inverse_factor =
      1.0 / (-M[0] * M[10] * M[13] * M[7] + M[0] * M[10] * M[15] * M[5] +
             M[0] * M[11] * M[13] * M[6] - M[0] * M[11] * M[14] * M[5] +
             M[0] * M[14] * M[7] * M[9] - M[0] * M[15] * M[6] * M[9] +
             M[10] * M[12] * M[1] * M[7] - M[10] * M[12] * M[3] * M[5] +
             M[10] * M[13] * M[3] * M[4] - M[10] * M[15] * M[1] * M[4] -
             M[11] * M[12] * M[1] * M[6] + M[11] * M[12] * M[2] * M[5] -
             M[11] * M[13] * M[2] * M[4] + M[11] * M[14] * M[1] * M[4] -
             M[12] * M[2] * M[7] * M[9] + M[12] * M[3] * M[6] * M[9] +
             M[13] * M[2] * M[7] * M[8] - M[13] * M[3] * M[6] * M[8] -
             M[14] * M[1] * M[7] * M[8] - M[14] * M[3] * M[4] * M[9] +
             M[14] * M[3] * M[5] * M[8] + M[15] * M[1] * M[6] * M[8] +
             M[15] * M[2] * M[4] * M[9] - M[15] * M[2] * M[5] * M[8]);
  L[0] = (-M[10] * M[13] * M[7] + M[10] * M[15] * M[5] + M[11] * M[13] * M[6] -
          M[11] * M[14] * M[5] + M[14] * M[7] * M[9] - M[15] * M[6] * M[9]) *
         inverse_factor;
  L[1] = (M[10] * M[13] * M[3] - M[10] * M[15] * M[1] - M[11] * M[13] * M[2] +
          M[11] * M[14] * M[1] - M[14] * M[3] * M[9] + M[15] * M[2] * M[9]) *
         inverse_factor;
  L[2] = (M[13] * M[2] * M[7] - M[13] * M[3] * M[6] - M[14] * M[1] * M[7] +
          M[14] * M[3] * M[5] + M[15] * M[1] * M[6] - M[15] * M[2] * M[5]) *
         inverse_factor;
  L[3] = (M[10] * M[1] * M[7] - M[10] * M[3] * M[5] - M[11] * M[1] * M[6] +
          M[11] * M[2] * M[5] - M[2] * M[7] * M[9] + M[3] * M[6] * M[9]) *
         inverse_factor;
  L[4] = (M[10] * M[12] * M[7] - M[10] * M[15] * M[4] - M[11] * M[12] * M[6] +
          M[11] * M[14] * M[4] - M[14] * M[7] * M[8] + M[15] * M[6] * M[8]) *
         inverse_factor;
  L[5] = (M[0] * M[10] * M[15] - M[0] * M[11] * M[14] - M[10] * M[12] * M[3] +
          M[11] * M[12] * M[2] + M[14] * M[3] * M[8] - M[15] * M[2] * M[8]) *
         inverse_factor;
  L[6] = (M[0] * M[14] * M[7] - M[0] * M[15] * M[6] - M[12] * M[2] * M[7] +
          M[12] * M[3] * M[6] - M[14] * M[3] * M[4] + M[15] * M[2] * M[4]) *
         inverse_factor;
  L[7] = (-M[0] * M[10] * M[7] + M[0] * M[11] * M[6] + M[10] * M[3] * M[4] -
          M[11] * M[2] * M[4] + M[2] * M[7] * M[8] - M[3] * M[6] * M[8]) *
         inverse_factor;
  L[8] = (M[11] * M[12] * M[5] - M[11] * M[13] * M[4] - M[12] * M[7] * M[9] +
          M[13] * M[7] * M[8] + M[15] * M[4] * M[9] - M[15] * M[5] * M[8]) *
         inverse_factor;
  L[9] = (M[0] * M[11] * M[13] - M[0] * M[15] * M[9] - M[11] * M[12] * M[1] +
          M[12] * M[3] * M[9] - M[13] * M[3] * M[8] + M[15] * M[1] * M[8]) *
         inverse_factor;
  L[10] = (-M[0] * M[13] * M[7] + M[0] * M[15] * M[5] + M[12] * M[1] * M[7] -
           M[12] * M[3] * M[5] + M[13] * M[3] * M[4] - M[15] * M[1] * M[4]) *
          inverse_factor;
  L[11] = (-M[0] * M[11] * M[5] + M[0] * M[7] * M[9] + M[11] * M[1] * M[4] -
           M[1] * M[7] * M[8] - M[3] * M[4] * M[9] + M[3] * M[5] * M[8]) *
          inverse_factor;
  L[12] = (-M[10] * M[12] * M[5] + M[10] * M[13] * M[4] + M[12] * M[6] * M[9] -
           M[13] * M[6] * M[8] - M[14] * M[4] * M[9] + M[14] * M[5] * M[8]) *
          inverse_factor;
  L[13] = (-M[0] * M[10] * M[13] + M[0] * M[14] * M[9] + M[10] * M[12] * M[1] -
           M[12] * M[2] * M[9] + M[13] * M[2] * M[8] - M[14] * M[1] * M[8]) *
          inverse_factor;
  L[14] = (M[0] * M[13] * M[6] - M[0] * M[14] * M[5] - M[12] * M[1] * M[6] +
           M[12] * M[2] * M[5] - M[13] * M[2] * M[4] + M[14] * M[1] * M[4]) *
          inverse_factor;
  L[15] = (M[0] * M[10] * M[5] - M[0] * M[6] * M[9] - M[10] * M[1] * M[4] +
           M[1] * M[6] * M[8] + M[2] * M[4] * M[9] - M[2] * M[5] * M[8]) *
          inverse_factor;
}

} // namespace NESO::Particles

#endif
