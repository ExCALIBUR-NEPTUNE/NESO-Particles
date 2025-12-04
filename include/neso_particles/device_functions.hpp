#ifndef _NESO_PARTICLES_DEVICE_FUNCTIONS_HPP_
#define _NESO_PARTICLES_DEVICE_FUNCTIONS_HPP_

#include "sycl_typedefs.hpp"
#include "typedefs.hpp"
#include <limits>

namespace NESO::Particles {

namespace Kernel {

template <typename T, typename U> inline auto min(const T &x, const U &y) {
  return KERNEL_MIN(x, y);
}
inline auto min(const REAL &x, const REAL &y) { return sycl::fmin(x, y); }
template <typename T, typename U> inline auto max(const T &x, const U &y) {
  return KERNEL_MAX(x, y);
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
template <typename T> inline auto pow(const T x, const T y) {
  return sycl::pow(x, y);
}
template <typename T> inline auto atan2(const T y, const T x) {
  return sycl::atan2(y, x);
}
template <typename T> inline auto rsqrt(const T x) { return sycl::rsqrt(x); }
template <typename T> inline auto sin(const T x) { return sycl::sin(x); }
template <typename T> inline auto cos(const T x) { return sycl::cos(x); }
template <typename T> inline auto tan(const T x) { return sycl::tan(x); }
template <typename T> inline auto log(const T x) { return sycl::log(x); }
template <typename T> inline auto log2(const T x) { return sycl::log2(x); }
template <typename T> inline auto log10(const T x) { return sycl::log10(x); }
template <typename T> inline auto round(const T x) { return sycl::round(x); }
template <typename T> inline auto tgamma(const T x) { return sycl::tgamma(x); }
template <typename T> inline auto trunc(const T x) { return sycl::trunc(x); }

namespace Private {
// ACPP does not seem to define a sycl::sincos(REAL, REAL*)
template <class, class = void> struct sincos_exists_for_t : std::false_type {};

template <class T>
struct sincos_exists_for_t<T, std::void_t<decltype(sycl::sincos(
                                  std::declval<T>(), std::declval<T *>))>>
    : std::true_type {};
} // namespace Private
template <typename T,
          std::enable_if_t<
              std::is_same<typename Private::sincos_exists_for_t<T>::type,
                           std::true_type>::value,
              bool> = true>
inline auto sincos(const T x, T *cosval) {
  return sycl::sincos(x, cosval);
}
template <typename T,
          std::enable_if_t<
              !std::is_same<typename Private::sincos_exists_for_t<T>::type,
                            std::true_type>::value,
              bool> = true>
inline auto sincos(const T x, T *cosval) {
  *cosval = sycl::cos(x);
  return sycl::sin(x);
}

template <typename T> inline auto dot_product_2d(const T *a, const T *b) {
  return a[0] * b[0] + a[1] * b[1];
}
template <typename T> inline auto dot_product_3d(const T *a, const T *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <int N, typename T> inline T dot_product(const T *a, const T *b) {
  T value = static_cast<T>(0);
  for (int ix = 0; ix < N; ix++) {
    value += a[ix] * b[ix];
  }
  return value;
}
template <typename T>
inline void cross_product(const T &a1, const T &a2, const T &a3, const T &b1,
                          const T &b2, const T &b3, T *RESTRICT c1,
                          T *RESTRICT c2, T *RESTRICT c3) {
  *c1 = ((a2) * (b3)) - ((a3) * (b2));
  *c2 = ((a3) * (b1)) - ((a1) * (b3));
  *c3 = ((a1) * (b2)) - ((a2) * (b1));
}

} // namespace Kernel

/**
 * For a line segment [ax, ay] - [bx, by] return ix,iy,jx,jy such that [ix, iy]
 * - [jx, jy] is the same line segment as [ax, ay] - [bx, by] but the outputs
 * are independent of the order in which a and b are specified. i.e. the
 * direction of the output line segment is always in the same direction.
 *
 * @param[in] ax First point of line segment, x coordinate.
 * @param[in] ay First point of line segment, y coordinate.
 * @param[in] bx Second point of line segment, x coordinate.
 * @param[in] by Second point of line segment, y coordinate.
 * @param[in, out] ix First point of output line segment, x coordinate.
 * @param[in, out] iy First point of output line segment, y coordinate.
 * @param[in, out] jx Second point of output line segment, x coordinate.
 * @param[in, out] jy Second point of output line segment, y coordinate.
 */
inline void consistent_line_orientation_2d(const REAL ax, const REAL ay,
                                           const REAL bx, const REAL by,
                                           REAL *ix, REAL *iy, REAL *jx,
                                           REAL *jy) {
  const bool axfirst = ax < bx;
  const bool ayfirst = ay < by;
  const bool afirst = ax == bx ? ayfirst : axfirst;
  *ix = afirst ? ax : bx;
  *iy = afirst ? ay : by;
  *jx = afirst ? bx : ax;
  *jy = afirst ? by : ay;
}

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
    const bool in_bounds_01 =
        ((0.0 - tol) <= l0_out) && (l0_out <= (1.0 + tol));

    // This segment is the one where we want strict bounds.
    const bool x_cond = (x0 < x1);
    const REAL x_min = x_cond ? x0 : x1;
    const REAL x_max = x_cond ? x1 : x0;
    const bool y_cond = (y0 < y1);
    const REAL y_min = y_cond ? y0 : y1;
    const REAL y_max = y_cond ? y1 : y0;

    const bool in_bounds_ab = (x_min - tol <= xi) && (xi <= x_max + tol) &&
                              (y_min - tol <= yi) && (yi <= y_max + tol);

    return in_bounds_01 && in_bounds_ab;
  } else {
    return false;
  }
}

/**
 * Compute the intersection point of two line segments
 * [(xa, ya), (xb, yb)] and [(x0, y0), (x1, y0)].
 *
 * Note that the second line segment, (x,y) is aligned with the x-axis and we
 * assume that !(ya == y0 && yb == y0).
 *
 * param[in] xa Input coordinate for point a.
 * param[in] ya Input coordinate for point a.
 * param[in] xb Input coordinate for point b.
 * param[in] yb Input coordinate for point b.
 * param[in] x0 Input coordinate for point x0.
 * param[in] y0 Input coordinate for point y0.
 * param[in] x1 Input coordinate for point x0.
 * param[in, out] xi Output intersection point if it exists.
 * param[in, out] yi Output intersection point if it exists.
 * param[in] tol Tolerance for intersection, e.g. how closely do the lines pass
 * at the ends default 0.0.
 * @returns True if the line segments intersect otherwise false.
 */
inline bool line_segment_intersection_2d_x_axis_aligned(
    const REAL &xa, const REAL &ya, const REAL &xb, const REAL &yb,
    const REAL &x0, const REAL &y0, const REAL &x1, REAL &xi, REAL &yi,
    const REAL tol = 0.0) {

  const REAL diff_a = ya - y0;
  const REAL diff_b = yb - y0;
  const bool is_crossed = (0 >= (diff_a * diff_b));

  const REAL abs_diff_a = Kernel::abs(diff_a);
  const REAL abs_diff_b = Kernel::abs(diff_b);
  const REAL width = (abs_diff_a + abs_diff_b);
  const bool colocated = (width == 0.0);
  const REAL ratio = (!colocated) ? abs_diff_a / width : 0.0;

  xi = xa + ratio * (xb - xa);
  yi = y0;

  const bool in_bounds = (((x0 - tol) <= xi) && (xi <= (x1 + tol))) ||
                         (((x1 - tol) <= xi) && (xi <= (x0 + tol)));

  return is_crossed && (!colocated) && in_bounds;
}

/**
 * Compute the intersection point of two line segments
 * [(xa, ya), (xb, yb)] and [(x0, y0), (x0, y1)].
 *
 * Note that the second line segment, (x,y) is aligned with the y-axis and we
 * assume that !(xa == x0 && xb == x0).
 *
 * param[in] xa Input coordinate for point a.
 * param[in] ya Input coordinate for point a.
 * param[in] xb Input coordinate for point b.
 * param[in] yb Input coordinate for point b.
 * param[in] x0 Input coordinate for point x0.
 * param[in] y0 Input coordinate for point y0.
 * param[in] y1 Input coordinate for point y0.
 * param[in, out] xi Output intersection point if it exists.
 * param[in, out] yi Output intersection point if it exists.
 * param[in] tol Tolerance for intersection, e.g. how closely do the lines pass
 * at the ends default 0.0.
 * @returns True if the line segments intersect otherwise false.
 */
inline bool line_segment_intersection_2d_y_axis_aligned(
    const REAL &xa, const REAL &ya, const REAL &xb, const REAL &yb,
    const REAL &x0, const REAL &y0, const REAL &y1, REAL &xi, REAL &yi,
    const REAL tol = 0.0) {
  return line_segment_intersection_2d_x_axis_aligned(ya, xa, yb, xb, y0, x0, y1,
                                                     yi, xi, tol);
}

/**
 * Intersection of line segment
 *
 *  [(ax, ay, az), (bx, by, bz)]
 *
 * with the plane segment
 *
 * (p0x, p2y, p0z)    (p1x, p2y, p0z)
 *               ------   y
 *              |      |  ^
 *              |      |  |
 *               ------   ---> x
 * (p0x, p0y, p0z)    (p1x, p0y, p0z)
 *
 * Assumes that the line does not lie in the plane.
 *
 * param[in] ax Input coordinate for point a.
 * param[in] ay Input coordinate for point a.
 * param[in] az Input coordinate for point a.
 * param[in] bx Input coordinate for point b.
 * param[in] by Input coordinate for point b.
 * param[in] bz Input coordinate for point b.
 * param[in] p0x Input coordinate for plane.
 * param[in] p0y Input coordinate for plane.
 * param[in] p0z Input coordinate for plane.
 * param[in] p1x Input coordinate for plane.
 * param[in] p2y Input coordinate for plane.
 * param[in, out] xi Output intersection point if it exists.
 * param[in, out] yi Output intersection point if it exists.
 * param[in, out] zi Output intersection point if it exists.
 * param[in] tol Tolerance for intersection, e.g. how closely do the lines pass
 * at the ends default 0.0.
 * @returns True if the line segments intersect otherwise false.
 */
inline bool plane_intersection_3d_xy_plane_aligned(
    const REAL &ax, const REAL &ay, const REAL &az, const REAL &bx,
    const REAL &by, const REAL &bz, const REAL &p0x, const REAL &p0y,
    const REAL &p0z, const REAL &p1x, const REAL &p2y, REAL &xi, REAL &yi,
    REAL &zi, const REAL tol = 0.0) {
  const REAL diff_a = az - p0z;
  const REAL diff_b = bz - p0z;
  const bool is_crossed = (0 >= (diff_a * diff_b));
  const REAL abs_diff_a = Kernel::abs(diff_a);
  const REAL abs_diff_b = Kernel::abs(diff_b);
  const REAL width = (abs_diff_a + abs_diff_b);
  const bool colocated = (width == 0.0);
  const REAL ratio = (!colocated) ? abs_diff_a / width : 0.0;

  xi = ax + ratio * (bx - ax);
  yi = ay + ratio * (by - ay);
  zi = p0z;

  const bool in_bounds_x = (((p0x - tol) <= xi) && (xi <= (p1x + tol))) ||
                           (((p1x - tol) <= xi) && (xi <= (p0x + tol)));
  const bool in_bounds_y = (((p0y - tol) <= yi) && (yi <= (p2y + tol))) ||
                           (((p2y - tol) <= yi) && (yi <= (p0y + tol)));

  return is_crossed && (!colocated) && in_bounds_x && in_bounds_y;
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

/**
 * Wrapper to perform fetch_max on an address and return the previous value,
 * i.e.
 *
 * max(current, value). This implementation uses strong CAS.
 *
 * @param ptr Address to contiainng current value.
 * @param value Value to perform operation with.
 * @returns Original value before new value was reduced.
 */
template <typename T>
inline T atomic_fetch_max_cas_strong(T *ptr, const T value) {

  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
      element_atomic(*ptr);
  // This implementation deliberately avoids issuing an atomic load.
  T expected = std::numeric_limits<T>::min();
  T desired;
  do {
    desired = sycl::max(value, expected);
  } while ((!element_atomic.compare_exchange_strong(expected, desired) &&
            (expected < value)));

  return expected;
}

/**
 * Wrapper to perform fetch_min on an address and return the previous value,
 * i.e.
 *
 * min(current, value). This implementation uses strong CAS.
 *
 * @param ptr Address to contiainng current value.
 * @param value Value to perform operation with.
 * @returns Original value before new value was reduced.
 */
template <typename T>
inline T atomic_fetch_min_cas_strong(T *ptr, const T value) {

  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
      element_atomic(*ptr);
  // This implementation deliberately avoids issuing an atomic load.
  T expected = std::numeric_limits<T>::max();
  T desired;
  do {
    desired = sycl::min(value, expected);
  } while ((!element_atomic.compare_exchange_strong(expected, desired) &&
            (expected > value)));

  return expected;
}

/**
 * Wrapper to perform fetch_add on an address and return the value.
 *
 * @param ptr Address to atomically increment.
 * @param value Value to increment by.
 * @returns Original value before new value was incremented.
 */
template <typename T> inline T atomic_fetch_add(T *ptr, const T value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
      a_ref(*ptr);
  return a_ref.fetch_add(value);
}

/**
 * Wrapper to perform fetch_min on an address and return the previous value,
 * i.e.
 *
 * min(current, value).
 *
 * @param ptr Address to contiainng current value.
 * @param value Value to perform operation with.
 * @returns Original value before new value was reduced.
 */
template <typename T> inline T atomic_fetch_min(T *ptr, const T value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
      a_ref(*ptr);
  return a_ref.fetch_min(value);
}

/**
 * Wrapper to perform fetch_max on an address and return the previous value,
 * i.e.
 *
 * max(current, value).
 *
 * @param ptr Address to contiainng current value.
 * @param value Value to perform operation with.
 * @returns Original value before new value was reduced.
 */
template <typename T> inline T atomic_fetch_max(T *ptr, const T value) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
      a_ref(*ptr);
  return a_ref.fetch_max(value);
}

// Are we using an AdaptiveCpp CUDA pass?
#ifdef __ACPP_ENABLE_CUDA_TARGET__
// Is this the nvcxx backend?
#ifdef __NVCOMPILER

inline INT atomic_fetch_min(INT *ptr, const INT value) {
  return atomic_fetch_min_cas_strong(ptr, value);
}
#define NESO_PARTICLES_CAS_MIN_INT
inline INT atomic_fetch_max(INT *ptr, const INT value) {
  return atomic_fetch_max_cas_strong(ptr, value);
}
#define NESO_PARTICLES_CAS_MAX_INT

#else // Assume that if we are not using the nvcxx backend then it is the clang
      // cuda backend.

inline REAL atomic_fetch_min(REAL *ptr, const REAL value) {
  return atomic_fetch_min_cas_strong(ptr, value);
}
#define NESO_PARTICLES_CAS_MIN_REAL
inline REAL atomic_fetch_max(REAL *ptr, const REAL value) {
  return atomic_fetch_max_cas_strong(ptr, value);
}
#define NESO_PARTICLES_CAS_MAX_REAL

inline INT atomic_fetch_min(INT *ptr, const INT value) {
  return atomic_fetch_min_cas_strong(ptr, value);
}
#define NESO_PARTICLES_CAS_MIN_INT
inline INT atomic_fetch_max(INT *ptr, const INT value) {
  return atomic_fetch_max_cas_strong(ptr, value);
}
#define NESO_PARTICLES_CAS_MAX_INT

#endif
#endif

namespace Kernel {

template <typename T> using plus = sycl::plus<T>;
template <typename T> using minimum = sycl::minimum<T>;
template <typename T> using maximum = sycl::maximum<T>;

template <typename T> constexpr T get_identity(sycl::plus<T>) {
  return static_cast<T>(0.0);
}
template <typename T> constexpr T get_identity(sycl::minimum<T>) {
  return static_cast<T>(std::numeric_limits<T>::max());
}
template <typename T> constexpr T get_identity(sycl::maximum<T>) {
  return static_cast<T>(std::numeric_limits<T>::lowest());
}

template <typename T>
inline void atomic_reduce(sycl::plus<T>, T *ptr, const T value) {
  atomic_fetch_add(ptr, value);
}
template <typename T>
inline void atomic_reduce(sycl::minimum<T>, T *ptr, const T value) {
  atomic_fetch_min(ptr, value);
}
template <typename T>
inline void atomic_reduce(sycl::maximum<T>, T *ptr, const T value) {
  atomic_fetch_max(ptr, value);
}

/**
 * Wrapper around joint_reduce for buggy implementations.
 *
 * @param group SYCL group.
 * @param d_first Pointer to first element.
 * @param d_last Pointer to element after last element.
 * @param binary_op Binary operation for reduction.
 * @returns Reduced value.
 */
template <typename GROUP_TYPE, typename VALUE_TYPE, typename OP_TYPE>
VALUE_TYPE joint_reduce(GROUP_TYPE group, VALUE_TYPE *d_first,
                        VALUE_TYPE *d_last, OP_TYPE binary_op) {
#ifdef __INTEL_LLVM_COMPILER
  return sycl::joint_reduce(group, d_first, d_last, binary_op);
#else
  VALUE_TYPE value = get_identity(binary_op);
  d_first += group.get_local_linear_id();
  while (d_first < d_last) {
    value = binary_op(value, *d_first);
    d_first += group.get_local_linear_range();
  }
  value = sycl::reduce_over_group(group, value, binary_op);
  return value;
#endif
}

/**
 * In place bitonic sort of 8 elements. See the following reference for more
 * details.
 *
 * A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel Skylake
 * DOI: 10.14569/IJACSA.2017.081044
 *
 * @param group SYCL group.
 * @param s_ptr kernel shared memory containing eight values to sort.
 */
template <typename GROUP_TYPE, typename VALUE_TYPE>
inline void bitonic8(GROUP_TYPE group, VALUE_TYPE *s_ptr) {
  const int i0 = static_cast<int>(group.get_local_id()) % 8;

  constexpr int indices[6][8] = {
      {1, 0, 3, 2, 5, 4, 7, 6}, {3, 2, 1, 0, 7, 6, 5, 4},
      {1, 0, 3, 2, 5, 4, 7, 6}, {7, 6, 5, 4, 3, 2, 1, 0},
      {2, 3, 0, 1, 6, 7, 4, 5}, {1, 0, 3, 2, 5, 4, 7, 6}};

  sycl::group_barrier(group);
  VALUE_TYPE v0 = s_ptr[i0];

  for (int stagex = 0; stagex < 6; stagex++) {
    const int i1 = indices[stagex][i0];

    sycl::group_barrier(group);
    const VALUE_TYPE v1 = s_ptr[i1];
    const VALUE_TYPE vmax = v1 < v0 ? v0 : v1;
    const VALUE_TYPE vmin = v1 < v0 ? v1 : v0;
    v0 = i0 < i1 ? vmin : vmax;

    sycl::group_barrier(group);
    s_ptr[i0] = v0;
  }
}

} // namespace Kernel

} // namespace NESO::Particles

#endif
