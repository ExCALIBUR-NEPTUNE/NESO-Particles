#ifndef _NESO_PARTICLES_DEVICE_FUNCTIONS_HPP_
#define _NESO_PARTICLES_DEVICE_FUNCTIONS_HPP_

#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

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
 * @returns True if the line segments intersect otherwise false.
 */
inline bool line_segment_intersection_2d(const REAL &xa, const REAL &ya,
                                         const REAL &xb, const REAL &yb,
                                         const REAL &x0, const REAL &y0,
                                         const REAL &x1, const REAL &y1,
                                         REAL &xi, REAL &yi, REAL &l0_out) {

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
      yi = xa + l0 * (xb - xa);
      l0_out = l0;
    } else {
      line_segment_intersection_2d_lambda(x0, y0, x1, y1, xa, ya, xb, yb, l0,
                                          l1);
      xi = x0 + l0 * (x1 - x0);
      yi = y0 + l0 * (y1 - y0);
      l0_out = l1;
    }
    return (0 <= l0) && (l0 <= 1.0) && (0 <= l1) && (l1 <= 1.0);
  } else {
    return false;
  }
}

} // namespace NESO::Particles

#endif
