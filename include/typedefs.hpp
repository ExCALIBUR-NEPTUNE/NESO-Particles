#ifndef _NESO_PARTICLES_TYPEDEFS
#define _NESO_PARTICLES_TYPEDEFS

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
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

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) < (y)) ? (y) : (x))
#define ABS(x) (((x) >= 0) ? (x) : (-(x)))
template <typename T>
void get_decomp_1d(const T N_compute_units, const T N_work_items,
                   const T work_unit, T *rstart, T *rend) {

  const auto pq = std::div(N_work_items, N_compute_units);
  const T i = work_unit;
  const T p = pq.quot;
  const T q = pq.rem;
  const T n = (i < q) ? (p + 1) : p;
  const T start = (MIN(i, q) * (p + 1)) + ((i > q) ? (i - q) * p : 0);
  const T end = start + n;

  *rstart = start;
  *rend = end;
}

#ifdef NESO_PARTICLES_DEVICE_TYPE_CPU

#define NESO_PARTICLES_DEVICE_LABEL "CPU"
#define NESO_PARTICLES_ITER_CELLS 1

#define NESO_PARTICLES_KERNEL_START                                            \
  const int neso_npart = pl_npart_cell[idx];                                   \
  for (int neso_layer = 0; neso_layer < neso_npart; neso_layer++) {
#define NESO_PARTICLES_KERNEL_END }
#define NESO_PARTICLES_KERNEL_CELL idx
#define NESO_PARTICLES_KERNEL_LAYER neso_layer

#else

#define NESO_PARTICLES_DEVICE_LABEL "GPU"
#define NESO_PARTICLES_ITER_PARTICLES 1

#define NESO_PARTICLES_KERNEL_START                                            \
  if ((((INT)idx) % pl_stride) < (pl_npart_cell[((INT)idx) / pl_stride])) {
#define NESO_PARTICLES_KERNEL_END }
#define NESO_PARTICLES_KERNEL_CELL (((INT)idx) / pl_stride)
#define NESO_PARTICLES_KERNEL_LAYER (((INT)idx) % pl_stride)

#endif

} // namespace NESO::Particles

#endif
