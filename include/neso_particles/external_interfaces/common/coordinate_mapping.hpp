#ifndef _NESO_PARTICLES_EXTERNAL_INTERFACES_COMMON_COORDINATE_MAPPING_H_
#define _NESO_PARTICLES_EXTERNAL_INTERFACES_COMMON_COORDINATE_MAPPING_H_
#include "../../device_functions.hpp"
#include "../../typedefs.hpp"

namespace NESO::Particles::ExternalCommon {

/**
 * Convert 2D Cartesian coordinates to Barycentric coordinates.
 *
 *  @param[in] x1 Triangle vertex 1, x component.
 *  @param[in] y1 Triangle vertex 1, y component.
 *  @param[in] x2 Triangle vertex 2, x component.
 *  @param[in] y2 Triangle vertex 2, y component.
 *  @param[in] x3 Triangle vertex 3, x component.
 *  @param[in] y3 Triangle vertex 3, y component.
 *  @param[in] x Point Cartesian coordinate, x component.
 *  @param[in] y Point Cartesian coordinate, y component.
 *  @param[in, out] l1 Point Barycentric coordinate, lambda 1.
 *  @param[in, out] l2 Point Barycentric coordinate, lambda 2.
 *  @param[in, out] l3 Point Barycentric coordinate, lambda 3.
 */
inline void
triangle_cartesian_to_barycentric(const REAL x1, const REAL y1, const REAL x2,
                                  const REAL y2, const REAL x3, const REAL y3,
                                  const REAL x, const REAL y, REAL *RESTRICT l1,
                                  REAL *RESTRICT l2, REAL *RESTRICT l3) {
  const REAL scaling = 1.0 / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
  *l1 = scaling * ((x2 * y3 - x3 * y2) + (y2 - y3) * x + (x3 - x2) * y);
  *l2 = scaling * ((x3 * y1 - x1 * y3) + (y3 - y1) * x + (x1 - x3) * y);
  *l3 = scaling * ((x1 * y2 - x2 * y1) + (y1 - y2) * x + (x2 - x1) * y);
};

/**
 * TODO
 */
inline void quad_reference_to_barycentric(

) {}

/**
 * TODO
 */
inline void quad_cartesian_to_collapsed(
    const REAL x0, const REAL y0, const REAL x1, const REAL y1, const REAL x2,
    const REAL y2, const REAL x3, const REAL y3, const REAL x, const REAL y,
    REAL *RESTRICT eta0, REAL *RESTRICT eta1) {
  const REAL a0 = 0.25 * (x0 + x1 + x2 + x3);
  const REAL a1 = 0.25 * (-x0 + x1 + x2 - x3);
  const REAL a2 = 0.25 * (-x0 - x1 + x2 + x3);
  const REAL a3 = 0.25 * (x0 - x1 + x2 - x3);
  const REAL b0 = 0.25 * (y0 + y1 + y2 + y3);
  const REAL b1 = 0.25 * (-y0 + y1 + y2 - y3);
  const REAL b2 = 0.25 * (-y0 - y1 + y2 + y3);
  const REAL b3 = 0.25 * (y0 - y1 + y2 - y3);

  // 0 = A * eta0^2 + B * eta0 + C
  const REAL A = a1 * b3 - a3 * b1;
  const REAL B = -x * b3 + a0 * b3 + a1 * b2 - a2 * b1 + a3 * y - a3 * b0;
  const REAL C = -x * b2 + a0 * b2 + a2 * y - a2 * b0;

  // Solve the quadratic in a numerically stable way
  const REAL determinate_inner = B * B - 4.0 * A * C;
  const REAL determinate =
      determinate_inner > 0.0 ? Kernel::sqrt(determinate_inner) : 0.0;
  const REAL i2A = 1.0 / (2.0 * A);
  const REAL Bpos = B >= 0.0;
  const REAL eta0p =
      Bpos ? ((-B - determinate) * i2A) : (2 * C) / (-B + determinate);
  const REAL eta0m =
      Bpos ? ((2.0 * C) / (-B - determinate)) : (-B + determinate) * i2A;

  // pick correct eta0, eta1 pair
  const REAL eta1p = (y - b0 - b1 * eta0p) / (b2 + b3 * eta0p);
  const REAL eta1m = (y - b0 - b1 * eta0m) / (b2 + b3 * eta0m);

  const REAL abs_eta0p = Kernel::abs(eta0p);
  const REAL abs_eta0m = Kernel::abs(eta0m);
  const REAL abs_eta1p = Kernel::abs(eta1p);
  const REAL abs_eta1m = Kernel::abs(eta1m);

  const bool containedp =
      ((abs_eta0p - 1.0) <= 0.0) && ((abs_eta1p - 1.0) <= 0.0);
  const bool containedm =
      ((abs_eta0m - 1.0) <= 0.0) && ((abs_eta1m - 1.0) <= 0.0);
  const REAL distp = (abs_eta0p - 1.0) + (abs_eta1p - 1.0);
  const REAL distm = (abs_eta0m - 1.0) + (abs_eta1m - 1.0);

  // closest pair
  const REAL eta0d = distp < distm ? eta0p : eta0m;
  const REAL eta1d = distp < distm ? eta1p : eta1m;
  // contained pair
  const REAL eta0c = containedp ? eta0p : eta0m;
  const REAL eta1c = containedp ? eta1p : eta1m;
  // If one of the pairs is contained then return the contained pair otherwise
  // select the closest pair
  *eta0 = containedp || containedm ? eta0c : eta0d;
  *eta1 = containedp || containedm ? eta1c : eta1d;
}

/**
 * TODO
 */
inline void quad_collapsed_to_cartesian(const REAL x0, const REAL y0,
                                        const REAL x1, const REAL y1,
                                        const REAL x2, const REAL y2,
                                        const REAL x3, const REAL y3,
                                        const REAL eta0, const REAL eta1,
                                        REAL *RESTRICT x, REAL *RESTRICT y) {
  const REAL w0 = 0.25 * (1.0 - eta0) * (1.0 - eta1);
  const REAL w1 = 0.25 * (1.0 + eta0) * (1.0 - eta1);
  const REAL w2 = 0.25 * (1.0 + eta0) * (1.0 + eta1);
  const REAL w3 = 0.25 * (1.0 - eta0) * (1.0 + eta1);
  *x = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
  *y = y0 * w0 + y1 * w1 + y2 * w2 + y3 * w3;
}

/**
 * TODO
 */
inline void quad_cartesian_to_barycentric(

) {}

/**
 * Convert 2D Barycentric coordinates to Cartesian coordinates.
 *
 *  @param[in] x1 Triangle vertex 1, x component.
 *  @param[in] y1 Triangle vertex 1, y component.
 *  @param[in] x2 Triangle vertex 2, x component.
 *  @param[in] y2 Triangle vertex 2, y component.
 *  @param[in] x3 Triangle vertex 3, x component.
 *  @param[in] y3 Triangle vertex 3, y component.
 *  @param[in] l1 Point Barycentric coordinate, lambda 1.
 *  @param[in] l2 Point Barycentric coordinate, lambda 2.
 *  @param[in] l3 Point Barycentric coordinate, lambda 3.
 *  @param[in, out] x Point Cartesian coordinate, x component.
 *  @param[in, out] y Point Cartesian coordinate, y component.
 */
inline void triangle_barycentric_to_cartesian(const REAL x1, const REAL y1,
                                              const REAL x2, const REAL y2,
                                              const REAL x3, const REAL y3,
                                              const REAL l1, const REAL l2,
                                              const REAL l3, REAL *RESTRICT x,
                                              REAL *RESTRICT y) {
  *x = l1 * x1 + l2 * x2 + l3 * x3;
  *y = l1 * y1 + l2 * y2 + l3 * y3;
};

/**
 * TODO
 */
inline void triangle_barycentric_invert(const REAL *RESTRICT L,
                                        REAL *RESTRICT M) {

  const REAL inverse_denom =
      1.0 / (L[0] * L[4] * L[8] - L[0] * L[5] * L[7] - L[1] * L[3] * L[8] +
             L[1] * L[5] * L[6] + L[2] * L[3] * L[7] - L[2] * L[4] * L[6]);

  // row: 0 col: 0
  M[0] = (L[4] * L[8] - L[5] * L[7]) * inverse_denom;
  // row: 0 col: 1
  M[1] = (-L[1] * L[8] + L[2] * L[7]) * inverse_denom;
  // row: 0 col: 2
  M[2] = (L[1] * L[5] - L[2] * L[4]) * inverse_denom;
  // row: 1 col: 0
  M[3] = (-L[3] * L[8] + L[5] * L[6]) * inverse_denom;
  // row: 1 col: 1
  M[4] = (L[0] * L[8] - L[2] * L[6]) * inverse_denom;
  // row: 1 col: 2
  M[5] = (-L[0] * L[5] + L[2] * L[3]) * inverse_denom;
  // row: 2 col: 0
  M[6] = (L[3] * L[7] - L[4] * L[6]) * inverse_denom;
  // row: 2 col: 1
  M[7] = (-L[0] * L[7] + L[1] * L[6]) * inverse_denom;
  // row: 2 col: 2
  M[8] = (L[0] * L[4] - L[1] * L[3]) * inverse_denom;
}

} // namespace NESO::Particles::ExternalCommon

#endif
