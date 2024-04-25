#ifndef _NESO_PARTICLES_SYCL_TYPEDEFS_H_
#define _NESO_PARTICLES_SYCL_TYPEDEFS_H_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
using namespace cl;
#endif

#endif
