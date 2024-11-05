#ifndef _NESO_PARTICLES_TYPEDEFS
#define _NESO_PARTICLES_TYPEDEFS

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

namespace NESO::Particles {

static inline int reduce_mul(const int nel, std::vector<int> &values) {
  int v = 1;
  for (int ex = 0; ex < nel; ex++) {
    v *= values[ex];
  }
  return v;
}

/**
 * \def NESOASSERT(expr, msg)
 * This is a helper macro to call the function neso_particles_assert. Users
 * should call this helper macro NESOASSERT like
 *
 *   NESOASSERT(conditional, message);
 *
 * To check conditionals within their code.
 */
#define NESOASSERT(expr, msg)                                                  \
  NESO::Particles::neso_particles_assert(#expr, expr, __FILE__, __LINE__, msg)

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
template <typename T>
inline void neso_particles_assert(const char *expr_str, bool expr,
                                  const char *file, int line, T &&msg) {
  if (!expr) {
    std::cerr << "NESO Particles Assertion error:\t" << msg << "\n"
              << "Expression evaluated:\t" << expr_str << "\n"
              << "Source location:\t\t" << file << ", line " << line << "\n";
#ifdef NESO_PARTICLES_NO_MPI_ABORT
    std::abort();
#else
    int flag = 0;
    MPI_Initialized(&flag);
    if (flag) {
      std::abort();
      MPI_Abort(MPI_COMM_WORLD, -1);
    } else {
      std::abort();
    }
#endif
  }
}

/**
 * \def NESOWARN(expr, msg)
 * This is a helper macro to call the function neso_particles_warn. Users
 * should call this helper macro NESOWARN like
 *
 *   NESOWARN(conditional, message);
 *
 * To check conditionals within their code.
 */
#ifdef NESO_PARTICLES_WARN
#define NESOWARN(expr, msg)                                                    \
  NESO::Particles::neso_particles_warn(#expr, expr, __FILE__, __LINE__, msg)
#else
#define NESOWARN(expr, msg)                                                    \
  {}
#endif

/**
 * This is a helper function to assert conditions are satisfied and print to
 * stderr if not. A warning is output on stderr. Users should call the
 * corresponding helper macro NESOWARN like
 *
 *   NESOWARN(conditional, message);
 *
 * To check conditionals within their code.
 *
 * @param expr_str A string identifying the conditional to check.
 * @param expr Bool resulting from the evaluation of the expression.
 * @param file Filename containing the call to neso_particles_assert.
 * @param line Line number for the call to neso_particles assert.
 * @param msg Message to print to stderr on evaluation of conditional to false.
 */
template <typename T>
inline void neso_particles_warn(const char *expr_str, bool expr,
                                const char *file, int line, T &&msg) {
  if (!expr) {
    std::cerr << "NESO Particles warning:\t" << msg << "\n"
              << "Expected value:\t" << expr_str << "\n"
              << "Source location:\t\t" << file << ", line " << line << "\n";
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

template <typename T>
inline T get_min_power_of_two(const T N_work_items, const size_t max_size) {
  const int base_two_power =
      static_cast<int>(std::log2(static_cast<double>(N_work_items)));
  const int base_two_power_p1 = base_two_power + 1;
  const int two_power_p1 = int(1) << base_two_power_p1;
  return (T)std::min(std::max((INT)two_power_p1, (INT)4), (INT)max_size);
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

#define nprint_variable(x) nprint(std::string(#x) + ":", x)

#ifndef NESO_PARTICLES_BLOCK_SIZE
#define NESO_PARTICLES_BLOCK_SIZE 1024
#endif

#ifdef NESO_PARTICLES_DEVICE_TYPE_CPU

#define NESO_PARTICLES_DEVICE_LABEL "CPU"
#define NESO_PARTICLES_ITER_CELLS 1

//#define NESO_PARTICLES_KERNEL_START                                            \
//  const int neso_npart = pl_npart_cell[idx];                                   \
//  for (int neso_layer = 0; neso_layer < neso_npart; neso_layer++) {
// #define NESO_PARTICLES_KERNEL_END }
// #define NESO_PARTICLES_KERNEL_CELL idx
// #define NESO_PARTICLES_KERNEL_LAYER neso_layer

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

// #define DEBUG_OOB_CHECK
#define DEBUG_OOB_WIDTH 1000

/**
 * Get the MPI thread level required by NESO-Particles. MPI should be
 * initialised by calling MPI_Init_thread with a required thread level greater
 * than or equal to the value returned by this function.
 *
 * @returns MPI thread level.
 */
inline int get_required_mpi_thread_level() { return MPI_THREAD_FUNNELED; }

/**
 * Test that a provided MPI thread level is sufficient for NESO-Particles.
 *
 * @param level Provided thread level.
 */
inline void test_provided_thread_level(const int level) {
  NESOASSERT(level >= get_required_mpi_thread_level(),
             "Provided MPI thread level is insufficient for NESO-Particles.");
}

/**
 * Helper function to initialise MPI and check that the provided thread level
 * is sufficient. Calling this function is equivalent to calling
 * MPI_Init_thread with the required thread level from
 * get_required_mpi_thread_level and checking the provided thread level is
 * equal to or greater than this required level.
 *
 * @param argc Pointer to the number of arguments.
 * @param argv Argument vector.
 */
inline void initialise_mpi(int *argc, char ***argv) {
  int provided_thread_level;
  NESOASSERT(MPI_Init_thread(argc, argv, get_required_mpi_thread_level(),
                             &provided_thread_level) == MPI_SUCCESS,
             "ERROR: MPI_Init_thread != MPI_SUCCESS");
  test_provided_thread_level(provided_thread_level);
}

inline std::string fixed_width_format(INT value) {
  char buffer[128];
  const int err = snprintf(buffer, 128, "% lld", static_cast<long long>(value));
  NESOASSERT(err >= 0 && err < 128, "Bad snprintf return code.");
  return std::string(buffer);
}
inline std::string fixed_width_format(REAL value) {
  char buffer[128];
  const int err = snprintf(buffer, 128, "% .6e", value);
  NESOASSERT(err >= 0 && err < 128, "Bad snprintf return code.");
  return std::string(buffer);
}
// TODO Move MPI typedefs to right place when dmplex branch merged.

/**
 * @returns True if device aware MPI is enabled.
 */
inline bool device_aware_mpi_enabled() {
#ifdef NESO_PARTICLES_DEVICE_AWARE_MPI
  constexpr bool enabled_cmake = true;
#else
  constexpr bool enabled_cmake = false;
#endif
  char *var_char;
  const bool var_exists =
      (var_char = std::getenv("NESO_PARTICLES_DEVICE_AWARE_MPI")) != nullptr;
  bool enabled = false;
  // If the env var exists then this sets if device MPI is enabled otherwise
  // use the cmake default
  if (var_exists) {
    std::string var_string = var_char;
    if ((var_string == "ON") || (var_string == "1")) {
      enabled = true;
    } else if ((var_string == "OFF") || (var_string == "0")) {
      enabled = false;
    } else {
      NESOASSERT(false, "Bad value for NESO_PARTICLES_DEVICE_AWARE_MPI, "
                        "expected ON, 1, OFF or 0. Value is: \"" +
                            var_string + "\".");
    }
  } else {
    enabled = enabled_cmake;
  }
  return enabled;
}

} // namespace NESO::Particles

// HDF5 includes if it exists.
#ifdef NESO_PARTICLES_HDF5
#include <hdf5.h>
#ifndef H5CHK
#define H5CHK(cmd) NESOASSERT((cmd) >= 0, "HDF5 ERROR");
#endif
#endif

#endif
