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
  //*l1 = scaling * ((x2 * y3 - x3 * y2) + (y2 - y3) * x + (x3 - x2) * y);
  //*l2 = scaling * ((x3 * y1 - x1 * y3) + (y3 - y1) * x + (x1 - x3) * y);
  //*l3 = scaling * ((x1 * y2 - x2 * y1) + (y1 - y2) * x + (x2 - x1) * y);
  *l1 = scaling * (Kernel::fma(x2, y3, -x3 * y2) +
                   Kernel::fma((y2 - y3), x, (x3 - x2) * y));
  *l2 = scaling * (Kernel::fma(x3, y1, -x1 * y3) +
                   Kernel::fma((y3 - y1), x, (x1 - x3) * y));
  *l3 = scaling * (Kernel::fma(x1, y2, -x2 * y1) +
                   Kernel::fma((y1 - y2), x, (x2 - x1) * y));
};

/**
 * Convert a collapsed reference coordinate into a Barycentric coordinate.
 *
 * @param[in] eta0 Collapsed coordinate, x component.
 * @param[in] eta1 Collapsed coordinate, y component.
 * @param[in, out] l0 Barycentric coordinate 0.
 * @param[in, out] l1 Barycentric coordinate 1.
 * @param[in, out] l2 Barycentric coordinate 2.
 * @param[in, out] l3 Barycentric coordinate 3.
 */
inline void quad_collapsed_to_barycentric(const REAL eta0, const REAL eta1,
                                          REAL *RESTRICT l0, REAL *RESTRICT l1,
                                          REAL *RESTRICT l2,
                                          REAL *RESTRICT l3) {
  const REAL etat0 = KERNEL_MAX(-1.0, KERNEL_MIN(1.0, eta0));
  const REAL etat1 = KERNEL_MAX(-1.0, KERNEL_MIN(1.0, eta1));
  const REAL xi0 = (etat0 + 1.0) * 0.5;
  const REAL xi1 = (etat1 + 1.0) * 0.5;

  *l0 = (1.0 - xi0) * (1.0 - xi1);
  *l1 = xi0 * (1.0 - xi1);
  *l2 = xi0 * xi1;
  *l3 = (1.0 - xi0) * xi1;
}

/**
 *  Map positions in physical space back to a reference (collapsed) coordinate
 *  system. Assumes that the vertices are ordered
 *
 *    v3 -- v2
 *    |     |
 *    v0 -- v1
 *
 *  and the the coordinate system is
 *
 *    (-1,  1) -- ( 1,  1)
 *        |           |
 *    (-1, -1) -- ( 1, -1)
 *
 *  Solves the bilinear system
 *
 *    xx = (1/4) (x0 (1-eta0)(1-eta1) + x1(1+eta0)(1-eta0)
 *                    + x2(1+eta0)(1+eta1) + x3(1-eta0)(1+eta1))
 *    yy = (1/4) (y0 (1-eta0)(1-eta1) + y1(1+eta0)(1-eta0)
 *                + y2(1+eta0)(1+eta1) + y3(1-eta0)(1+eta1))
 *
 *  for eta0 and eta1 by writing
 *
 *    a0 = 0.25 * (x0 + x1 + x2 + x3)
 *    a1 = 0.25 * (-x0 + x1 + x2 - x3)
 *    a2 = 0.25 * (-x0 - x1 + x2 + x3)
 *    a3 = 0.25 * (x0 - x1 + x2 - x3)
 *    b0 = 0.25 * (y0 + y1 + y2 + y3)
 *    b1 = 0.25 * (-y0 + y1 + y2 - y3)
 *    b2 = 0.25 * (-y0 - y1 + y2 + y3)
 *    b3 = 0.25 * (y0 - y1 + y2 - y3)
 *
 *  then rewriting the original equations as
 *
 *    xx = a0 + a1 eta0 + a2 eta1 + a3 eta0 eta1 (1)
 *    yy = b0 + b1 eta0 + b2 eta1 + b3 eta0 eta1
 *
 *  Now write
 *
 *    eta1 = (y - b0 - b1 eta0) / (b2 + b3 eta0) (2)
 *
 *  and subsitute eta1 in (1) for (2). The resulting quadratic should be
 *
 *    0 = A eta0^2 + B eta0 + C (3)
 *
 *  where
 *
 *    A = a1 b3 - a3 b1
 *    B = -x b3 + a0 b3 + a1 b2 - a2 b1 + a3 y - a3 b0
 *    C = -x b2 + a0 b2 + a2 y - a2 b0
 *
 *  Solve (3) to get eta0 then use (2) to compute eta1. If b2=0 and b3=0 then
 *  swap the equations and carry out the same process.
 *
 * @param[in] x0 First vertex x component.
 * @param[in] y0 First vertex y component.
 * @param[in] x1 Second vertex x component.
 * @param[in] y1 Second vertex y component.
 * @param[in] x2 Third vertex x component.
 * @param[in] y2 Third vertex y component.
 * @param[in] x3 Fourth vertex x component.
 * @param[in] y3 Fourth vertex y component.
 * @param[in] xx Point to map into reference space, x component.
 * @param[in] yy Point to map into reference space, y component.
 * @param[in, out] eta0 Collapsed coordinate, x component.
 * @param[in, out] eta1 Collapsed coordinate, y component.
 */
inline void quad_cartesian_to_collapsed(
    const REAL x0, const REAL y0, const REAL x1, const REAL y1, const REAL x2,
    const REAL y2, const REAL x3, const REAL y3, const REAL xx, const REAL yy,
    REAL *RESTRICT eta0, REAL *RESTRICT eta1) {

  const REAL aa0 = 0.25 * (x0 + x1 + x2 + x3);
  const REAL aa1 = 0.25 * (-x0 + x1 + x2 - x3);
  const REAL aa2 = 0.25 * (-x0 - x1 + x2 + x3);
  const REAL aa3 = 0.25 * (x0 - x1 + x2 - x3);
  const REAL bb0 = 0.25 * (y0 + y1 + y2 + y3);
  const REAL bb1 = 0.25 * (-y0 + y1 + y2 - y3);
  const REAL bb2 = 0.25 * (-y0 - y1 + y2 + y3);
  const REAL bb3 = 0.25 * (y0 - y1 + y2 - y3);

  // Do we need to swap the rows of the vector system to avoid a 0/0?
  const bool swap_rows = (Kernel::abs(bb2) + Kernel::abs(bb3)) <
                         (Kernel::abs(aa2) + Kernel::abs(aa3));
  const REAL a0 = swap_rows ? bb0 : aa0;
  const REAL a1 = swap_rows ? bb1 : aa1;
  const REAL a2 = swap_rows ? bb2 : aa2;
  const REAL a3 = swap_rows ? bb3 : aa3;
  const REAL b0 = swap_rows ? aa0 : bb0;
  const REAL b1 = swap_rows ? aa1 : bb1;
  const REAL b2 = swap_rows ? aa2 : bb2;
  const REAL b3 = swap_rows ? aa3 : bb3;
  const REAL x = swap_rows ? yy : xx;
  const REAL y = swap_rows ? xx : yy;

  // 0 = A * eta0^2 + B * eta0 + C
  // const REAL A = a1 * b3 - a3 * b1;
  const REAL A = Kernel::fma(a1, b3, -a3 * b1);
  // const REAL B = -x * b3 + a0 * b3 + a1 * b2 - a2 * b1 + a3 * y - a3 * b0;
  const REAL B = Kernel::fma(-x, b3, a0 * b3) + Kernel::fma(a1, b2, -a2 * b1) +
                 Kernel::fma(a3, y, -a3 * b0);
  // const REAL C = -x * b2 + a0 * b2 + a2 * y - a2 * b0;
  const REAL C = Kernel::fma(-x, b2, a0 * b2) + Kernel::fma(a2, y, -a2 * b0);

  // Solve the quadratic in a numerically stable way
  // const REAL determinate_inner = B * B - 4.0 * A * C;
  const REAL determinate_inner = Kernel::fma(-4.0 * A, C, B * B);
  const REAL determinate =
      determinate_inner > 0.0 ? Kernel::sqrt(determinate_inner) : 0.0;
  const REAL i2A = 1.0 / (2.0 * A);
  const bool bad2A = 2.0 * A == 0.0;
  const REAL Bpos = B >= 0.0;
  REAL eta0p = Bpos ? ((-B - determinate) * i2A) : (2 * C) / (-B + determinate);
  REAL eta0m =
      Bpos ? ((2.0 * C) / (-B - determinate)) : (-B + determinate) * i2A;

  // Determine if there are NaNs
  const bool bad_eta0p = Bpos ? bad2A : ((-B + determinate) == 0.0);
  const bool bad_eta0m = Bpos ? ((-B - determinate) == 0.0) : bad2A;

  const REAL denom1p = Kernel::fma(b3, eta0p, b2);
  const REAL denom1m = Kernel::fma(b3, eta0m, b2);
  const bool bad_eta1p = (denom1p == 0.0) || (bad_eta0p);
  const bool bad_eta1m = (denom1m == 0.0) || (bad_eta0m);

  // pick correct eta0, eta1 pair
  // const REAL eta1p = (y - b0 - b1 * eta0p) / (b2 + b3 * eta0p);
  REAL eta1p = Kernel::fma(-b1, eta0p, y - b0) / denom1p;
  // const REAL eta1m = (y - b0 - b1 * eta0m) / (b2 + b3 * eta0m);
  REAL eta1m = Kernel::fma(-b1, eta0m, y - b0) / denom1m;

  eta0p = (bad_eta0p) ? 1000.0 : eta0p;
  eta1p = (bad_eta1p) ? 1000.0 : eta1p;
  eta0m = (bad_eta0m) ? 1000.0 : eta0m;
  eta1m = (bad_eta1m) ? 1000.0 : eta1m;

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
 * Map a collapsed coordinate back to the physical coordinate.
 *
 * @param[in] x0 First vertex x component.
 * @param[in] y0 First vertex y component.
 * @param[in] x1 Second vertex x component.
 * @param[in] y1 Second vertex y component.
 * @param[in] x2 Third vertex x component.
 * @param[in] y2 Third vertex y component.
 * @param[in] x3 Fourth vertex x component.
 * @param[in] y3 Fourth vertex y component.
 * @param[in] eta0 Collapsed coordinate, x component.
 * @param[in] eta1 Collapsed coordinate, y component.
 * @param[in, out] x Coordinate in physical space, x component.
 * @param[in, out] y Coordinate in physical space, y component.
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
  // *x = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
  *x = Kernel::fma(x0, w0, x1 * w1) + Kernel::fma(x2, w2, x3 * w3);
  //*y = y0 * w0 + y1 * w1 + y2 * w2 + y3 * w3;
  *y = Kernel::fma(y0, w0, y1 * w1) + Kernel::fma(y2, w2, y3 * w3);
}

/**
 * Map a physical coordinate to Barycentric coordinates.
 *
 * @param[in] x0 First vertex x component.
 * @param[in] y0 First vertex y component.
 * @param[in] x1 Second vertex x component.
 * @param[in] y1 Second vertex y component.
 * @param[in] x2 Third vertex x component.
 * @param[in] y2 Third vertex y component.
 * @param[in] x3 Fourth vertex x component.
 * @param[in] y3 Fourth vertex y component.
 * @param[in] xx Point to map into reference space, x component.
 * @param[in] yy Point to map into reference space, y component.
 * @param[in, out] l0 Barycentric coordinate 0.
 * @param[in, out] l1 Barycentric coordinate 1.
 * @param[in, out] l2 Barycentric coordinate 2.
 * @param[in, out] l3 Barycentric coordinate 3.
 */
inline void quad_cartesian_to_barycentric(
    const REAL x0, const REAL y0, const REAL x1, const REAL y1, const REAL x2,
    const REAL y2, const REAL x3, const REAL y3, const REAL x, const REAL y,
    REAL *RESTRICT l0, REAL *RESTRICT l1, REAL *RESTRICT l2,
    REAL *RESTRICT l3) {
  REAL eta0, eta1;
  quad_cartesian_to_collapsed(x0, y0, x1, y1, x2, y2, x3, y3, x, y, &eta0,
                              &eta1);
  quad_collapsed_to_barycentric(eta0, eta1, l0, l1, l2, l3);
}

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
  *x = Kernel::fma(l1, x1, l2 * x2) + l3 * x3;
  *y = Kernel::fma(l1, y1, l2 * y2) + l3 * y3;
};

/**
 * Given a torus with major radius R convert the coordinate for a point
 * specified in (x,y,z) Cartesian coordinates to a torus cylindrical coordinate
 * system (r, theta, phi). In this notation:
 *
 *    z          /----\
 *    ^         /   /  \  In the (z, R) plane, radius "r", angle "theta"
 *    |        |   *----| relative to a.
 *    |         \  a   /
 *    |     R    \----/
 *    * ---------->
 *    ^
 *    |
 *  (0,0,0) In Cartesian.
 *
 *  @param[in] R Major radius of torus.
 *  @param[in] x Cartesian coordinate, x component.
 *  @param[in] y Cartesian coordinate, y component.
 *  @param[in] z Cartesian coordinate, z component.
 *  @param[in, out] r Torus cylindrical coordinate, radius in poloidal plane
 * from centre (a).
 *  @param[in, out] phi Torus cylindrical coordinate, toroidal angle relative to
 * origin (Cartesian (0,0,0)).
 *  @param[in, out] theta Torus cylindrical coordinate, angle in poloidal plane
 * from centre (a).
 */
inline void cartesian_to_torus_cylindrical(const REAL R, const REAL x,
                                           const REAL y, const REAL z, REAL *r,
                                           REAL *phi, REAL *theta) {
  *phi = Kernel::atan2(y, x);

  const REAL plane_radius2 = x * x + y * y;
  const REAL inverse_plane_radius = Kernel::rsqrt(plane_radius2);
  const REAL x_normalised = x * inverse_plane_radius;
  const REAL y_normalised = y * inverse_plane_radius;

  // Origin in poloidal plane is at (R, phi) = Cartesian (R * cos(theta), R *
  // sin(theta)))
  //                                         = Cartesian (R * x_normalised, R *
  //                                         y_normalised)
  const REAL po_x = R * x_normalised;
  const REAL po_y = R * y_normalised;
  const REAL px = x - po_x;
  const REAL py = y - po_y;

  // (px, py, z) . (x_normalised, y_normalised, 0) =
  //    px * x_normalised + py * y_normalised
  const REAL xplane = px * x_normalised + py * y_normalised;
  const REAL yplane = z;

  *r = Kernel::sqrt(px * px + py * py + z * z);
  *theta = Kernel::atan2(yplane, xplane);
}

/**
 * Given a torus with major radius R convert the coordinate for a point
 * specified in torus cylindrical coordinate system (r, theta, phi) to Cartesian
 * coordinates (x,y,z). In this notation:
 *
 *    z          /----\
 *    ^         /   /  \  In the (z, R) plane, radius "r", angle "theta"
 *    |        |   *----| relative to a.
 *    |         \  a   /
 *    |     R    \----/
 *    * ---------->
 *    ^
 *    |
 *  (0,0,0) In Cartesian.
 *
 *  @param[in] R Major radius of torus.
 *  @param[in] r Torus cylindrical coordinate, radius in poloidal plane
 * from centre (a).
 *  @param[in] phi Torus cylindrical coordinate, toroidal angle relative to
 * origin (Cartesian (0,0,0)).
 *  @param[in] theta Torus cylindrical coordinate, angle in poloidal plane
 * from centre (a).
 *  @param[in, out] x Cartesian coordinate, x component.
 *  @param[in, out] y Cartesian coordinate, y component.
 *  @param[in, out] z Cartesian coordinate, z component.
 *
 */
inline void torus_cylindrical_to_cartesian(const REAL R, const REAL r,
                                           const REAL phi, const REAL theta,
                                           REAL *x, REAL *y, REAL *z) {

  REAL costheta;
  const REAL sintheta = Kernel::sincos(theta, &costheta);

  *z = r * sintheta;

  REAL cosphi;
  const REAL sinphi = Kernel::sincos(phi, &cosphi);

  // The vector (planex/length, planey/length, 0) = (cos(phi), sin(phi), 0)
  // forms the "x" vector in the poloidol plane.
  const REAL planex = R * cosphi;
  const REAL planey = R * sinphi;

  *x = planex + r * costheta * cosphi;
  *y = planey + r * costheta * sinphi;
}

} // namespace NESO::Particles::ExternalCommon

#endif
