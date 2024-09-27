#ifndef _NESO_PARTICLES_UTILITY
#define _NESO_PARTICLES_UTILITY

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "access.hpp"
#include "compute_target.hpp"
#include "mesh_interface.hpp"
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

/**
 * Helper function to quickly initialise a uniform distribution of particles on
 * a CartesianHMesh.
 *
 * @param[in] mesh CartesianHMesh on which to spawn particles.
 * @param[in] npart_per_cell Number of particle positions to sample for each
 * mesh cell.
 * @param[in, out] positions Ouput particle positions indexed by dimension then
 * particle.
 * @param[in, out] cells Ouput particle cell ids indexed by particle.
 * @param[in] rng_in Optional input RNG to use.
 */
inline void uniform_within_cartesian_cells(
    CartesianHMeshSharedPtr mesh, const int npart_per_cell,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    std::optional<std::mt19937> rng_in = std::nullopt) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }
  const int ndim = mesh->get_ndim();
  std::vector<double> extents(ndim);

  const double cell_width_fine = mesh->get_cell_width_fine();
  for (int dx = 0; dx < ndim; dx++) {
    extents.at(dx) = cell_width_fine;
  }
  const int cell_count = mesh->get_cart_cell_count();
  const int npart_total = npart_per_cell * cell_count;
  positions.resize(ndim);
  cells.resize(npart_total);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(npart_total);
  }
  std::vector<double> origin(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    origin.at(dx) = mesh->get_mesh_hierarchy()->origin.at(dx);
  }

  auto mesh_cells = mesh->get_owned_cells();
  const bool single_cell_mode = mesh->single_cell_mode;

  for (int cx = 0; cx < cell_count; cx++) {
    const int index_start = cx * npart_per_cell;
    const int index_end = (cx + 1) * npart_per_cell;
    for (int ex = index_start; ex < index_end; ex++) {
      cells.at(ex) = single_cell_mode ? 0 : cx;
    }
    auto positions_ref_cell =
        uniform_within_extents(npart_per_cell, ndim, extents.data(), rng);

    std::vector<double> offset(ndim);
    for (int dx = 0; dx < ndim; dx++) {
      offset.at(dx) =
          origin.at(dx) + mesh_cells.at(cx).at(dx) * cell_width_fine;
    }

    int index = 0;
    for (int ex = index_start; ex < index_end; ex++) {
      for (int dx = 0; dx < ndim; dx++) {
        positions.at(dx).at(ex) =
            offset.at(dx) + positions_ref_cell.at(dx).at(index);
      }
      index++;
    }
  }
}

} // namespace NESO::Particles

#endif
