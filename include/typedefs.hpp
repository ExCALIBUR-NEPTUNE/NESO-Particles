#ifndef _NESO_PARTICLES_TYPEDEFS
#define _NESO_PARTICLES_TYPEDEFS

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
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
  neso_particles_assert(#expr, expr, __FILE__, __LINE__, msg)

/**
 * This is a helper function to assert conditions are satisfied and terminate
 * execution if not. An error is output on stderr and MPI_Abort is called if
 * MPI is initialised. Users should call the corresponding helper macro
 * NESOASSERT like
 *
 *   NESOASSERT(conditional, message);
 *
 * To check conditionals within their code.
 *
 * @param expr_str A string identifying the conditional to check.
 * @param expr Bool resulting from the evaluation of the expression.
 * @param file Filename containing the call to neso_particles_assert.
 * @param line Line number for the call to neso_particles assert.
 * @param msg Message to print to stderr on evaluation of conditional to false.
 */
inline void neso_particles_assert(const char *expr_str, bool expr,
                                  const char *file, int line, const char *msg) {
  if (!expr) {
    std::cerr << "NESO Particles Assertion error:\t" << msg << "\n"
              << "Expected value:\t" << expr_str << "\n"
              << "Source location:\t\t" << file << ", line " << line << "\n";
    int flag = 0;
    MPI_Initialized(&flag);
    if (flag) {
      MPI_Abort(MPI_COMM_WORLD, -1);
    } else {
      std::abort();
    }
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

template <typename U> inline void nprint_recurse(int flag, U next) {
  if (flag) {
    std::cout << " ";
  }
  std::cout << next << std::endl;
}

template <typename U, typename... T>
inline void nprint_recurse(int flag, U next, T... args) {
  if (flag) {
    std::cout << " ";
  }
  std::cout << next;
  nprint_recurse(++flag, args...);
}

template <typename... T> inline void nprint(T... args) {
  nprint_recurse(0, args...);
}

#ifndef NESO_PARTICLES_BLOCK_SIZE
#define NESO_PARTICLES_BLOCK_SIZE 1024
#endif

#ifdef NESO_PARTICLES_DEVICE_TYPE_CPU

#define NESO_PARTICLES_DEVICE_LABEL "CPU"
#define NESO_PARTICLES_ITER_CELLS 1

//#define NESO_PARTICLES_KERNEL_START                                            \
//  const int neso_npart = pl_npart_cell[idx];                                   \
//  for (int neso_layer = 0; neso_layer < neso_npart; neso_layer++) {
//#define NESO_PARTICLES_KERNEL_END }
//#define NESO_PARTICLES_KERNEL_CELL idx
//#define NESO_PARTICLES_KERNEL_LAYER neso_layer

#define NESO_PARTICLES_KERNEL_START                                            \
  const int neso_cell = (((INT)idx) / pl_stride);                              \
  const int neso_npart = pl_npart_cell[neso_cell];                             \
  const int neso_layer_start =                                                 \
      (((INT)idx) % pl_stride) * NESO_PARTICLES_BLOCK_SIZE;                    \
  const int neso_layer_end =                                                   \
      MIN(neso_layer_start + NESO_PARTICLES_BLOCK_SIZE, neso_npart);           \
  for (int neso_layer = neso_layer_start; neso_layer < neso_layer_end;         \
       neso_layer++) {

#define NESO_PARTICLES_KERNEL_END }
#define NESO_PARTICLES_KERNEL_CELL (((INT)idx) / pl_stride)
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

//#define DEBUG_OOB_CHECK
#define DEBUG_OOB_WIDTH 1000

} // namespace NESO::Particles

#endif
