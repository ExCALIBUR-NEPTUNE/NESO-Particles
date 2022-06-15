#ifndef _NESO_PARTICLES_TYPEDEFS
#define _NESO_PARTICLES_TYPEDEFS

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#define RESTRICT __restrict

namespace NESO::Particles {

static inline int reduce_mul(const int nel, std::vector<int> &values) {
  int v = 1;
  for (int ex = 0; ex < nel; ex++) {
    v *= values[ex];
  }
  return v;
}

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

template <typename T>
inline std::vector<size_t> reverse_argsort(const std::vector<T> &array) {
  std::vector<size_t> indices(array.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&array](int left, int right) -> bool {
              return array[left] > array[right];
            });

  return indices;
}

} // namespace NESO::Particles

#endif
