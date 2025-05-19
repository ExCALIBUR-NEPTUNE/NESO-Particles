#ifndef _NESO_PARTICLES_UTILITY
#define _NESO_PARTICLES_UTILITY

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "access.hpp"
#include "cartesian_mesh/cartesian_h_mesh.hpp"
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
std::vector<std::vector<double>>
uniform_within_extents(const int N, const int ndim, const double *extents);

/**
 *  Create (N)x(ndim) set of samples from a Gaussian distribution.
 *
 *  @param N Number of points to generate.
 *  @param ndim Number of dimensions.
 *  @param mu Mean to use for Gaussian distribution.
 *  @param sigma Sigma to use for Gaussian distribution.
 *  @returns (N)x(ndim) set of samples stored per column.
 */
std::vector<std::vector<double>> normal_distribution(const int N,
                                                     const int ndim,
                                                     const double mu,
                                                     const double sigma);

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
void uniform_within_cartesian_cells(
    CartesianHMeshSharedPtr mesh, const int npart_per_cell,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    std::optional<std::mt19937> rng_in = std::nullopt);

/**
 *  Get the total number of particles. This method must be called collectively
 * on the communicator.
 *
 * @param particle_group ParticleGroup or ParticleSubGroup to sum local particle
 * counts of.
 * @returns Global particle count.
 */
template <typename GROUP_TYPE>
inline INT get_npart_global(std::shared_ptr<GROUP_TYPE> particle_group) {
  const INT npart_local = static_cast<INT>(particle_group->get_npart_local());
  INT npart_global = 0;
  MPI_Comm comm =
      get_particle_group(particle_group)->sycl_target->comm_pair.comm_parent;

  MPICHK(MPI_Allreduce(&npart_local, &npart_global, 1,
                       map_ctype_mpi_type<INT>(), MPI_SUM, comm));

  return npart_global;
}

} // namespace NESO::Particles

#endif
