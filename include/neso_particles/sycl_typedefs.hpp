#ifndef _NESO_PARTICLES_SYCL_TYPEDEFS_H_
#define _NESO_PARTICLES_SYCL_TYPEDEFS_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
using namespace cl;
#endif

#ifndef KERNEL_MIN
#define KERNEL_MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef KERNEL_MAX
#define KERNEL_MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef KERNEL_ABS
#define KERNEL_ABS(x) (((x) < 0) ? (-(x)) : (x))
#endif

#ifndef KERNEL_CROSS_PRODUCT_3D
#define KERNEL_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)            \
  c1 = ((a2) * (b3)) - ((a3) * (b2));                                          \
  c2 = ((a3) * (b1)) - ((a1) * (b3));                                          \
  c3 = ((a1) * (b2)) - ((a2) * (b1));
#endif

#ifndef KERNEL_DOT_PRODUCT_3D
#define KERNEL_DOT_PRODUCT_3D(a1, a2, a3, b1, b2, b3)                          \
  ((a1) * (b1) + (a2) * (b2) + (a3) * (b3))
#endif

#ifndef KERNEL_DOT_PRODUCT_2D
#define KERNEL_DOT_PRODUCT_2D(a1, a2, b1, b2) ((a1) * (b1) + (a2) * (b2))
#endif

#endif
