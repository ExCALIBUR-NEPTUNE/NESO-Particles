#ifndef _NESO_PARTICLES_UTILITY
#define _NESO_PARTICLES_UTILITY

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "access.hpp"
#include "compute_target.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  Create a uniform distribution of particle positions within a set of extents.
 *
 *  @param N Number of points to generate.
 *  @param ndim Number of dimensions.
 *  @param extents Extent of each of the dimensions.
 *  @param rng RNG to use.
 *  @returns (N)x(ndim) set of positions stored for each column.
 */
template <typename RNG>
inline std::vector<std::vector<double>>
uniform_within_extents(const int N, const int ndim, const double *extents,
                       RNG &rng) {

  std::uniform_real_distribution<double> uniform_rng(0.0, 1.0);
  std::vector<std::vector<double>> positions(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(N);
    const double ex = extents[dimx];
    for (int px = 0; px < N; px++) {
      positions[dimx][px] = ex * uniform_rng(rng);
    }
  }

  return positions;
}

/**
 *  Create (N)x(ndim) set of samples from a Gaussian distribution.
 *
 *  @param N Number of points to generate.
 *  @param ndim Number of dimensions.
 *  @param mu Mean to use for Gaussian distribution.
 *  @param sigma Sigma to use for Gaussian distribution.
 *  @param rng RNG to use.
 *  @returns (N)x(ndim) set of samples stored per column.
 */
template <typename RNG>
inline std::vector<std::vector<double>>
normal_distribution(const int N, const int ndim, const double mu,
                    const double sigma, RNG &rng) {

  std::normal_distribution<> d{mu, sigma};
  std::vector<std::vector<double>> array(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    array[dimx] = std::vector<double>(N);
    for (int px = 0; px < N; px++) {
      array[dimx][px] = d(rng);
    }
  }

  return array;
}

/**
 *  Create a uniform distribution of particle positions within a set of extents.
 *
 *  @param N Number of points to generate.
 *  @param ndim Number of dimensions.
 *  @param extents Extent of each of the dimensions.
 *  @returns (N)x(ndim) set of positions stored for each column.
 */
inline std::vector<std::vector<double>>
uniform_within_extents(const int N, const int ndim, const double *extents) {
  std::mt19937 rng = std::mt19937(std::random_device{}());
  return uniform_within_extents(N, ndim, extents, rng);
}

/**
 *  Create (N)x(ndim) set of samples from a Gaussian distribution.
 *
 *  @param N Number of points to generate.
 *  @param ndim Number of dimensions.
 *  @param mu Mean to use for Gaussian distribution.
 *  @param sigma Sigma to use for Gaussian distribution.
 *  @returns (N)x(ndim) set of samples stored per column.
 */
inline std::vector<std::vector<double>>
normal_distribution(const int N, const int ndim, const double mu,
                    const double sigma) {
  std::mt19937 rng = std::mt19937(std::random_device{}());
  return normal_distribution(N, ndim, mu, sigma, rng);
}

} // namespace NESO::Particles

#endif
