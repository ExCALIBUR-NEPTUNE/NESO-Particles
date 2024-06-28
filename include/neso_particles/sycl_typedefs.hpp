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

#endif
