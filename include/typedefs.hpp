#ifndef _NESO_PARTICLES_TYPEDEFS
#define _NESO_PARTICLES_TYPEDEFS

#include <cstdint>
#include <iostream>

#define RESTRICT __restrict

namespace NESO::Particles {

#define NESOASSERT(expr, msg)                                                  \
  neso_particle_assert(#expr, expr, __FILE__, __LINE__, msg)

inline void neso_particle_assert(const char *expr_str, bool expr,
                                 const char *file, int line, const char *msg) {
  if (!expr) {
    std::cerr << "NESO Particles Assertion error:\t" << msg << "\n"
              << "Expected value:\t" << expr_str << "\n"
              << "Source location:\t\t" << file << ", line " << line << "\n";
    abort();
  }
}

typedef double REAL;
typedef int64_t INT;

} // namespace NESO::Particles

#endif
