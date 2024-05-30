#ifndef _NESO_PARTICLES_DMPLEX_UTILITY_HPP_
#define _NESO_PARTICLES_DMPLEX_UTILITY_HPP_

#include "dmplex_cell_serialise.hpp"
#include "dmplex_interface.hpp"
#include "petsc_common.hpp"
#include <petscviewerhdf5.h>
#include <string>

namespace NESO::Particles::PetscInterface {

/**
 * Helper function to quickly initialise a uniform distribution of particles on
 * a DMPlex mesh.
 *
 * @param[in] mesh DMPlexInterface on which to spawn particles.
 * @param[in] npart_per_cell Number of particle positions to sample for each
 * mesh cell.
 * @param[in, out] positions Ouput particle positions indexed by dimension then
 * particle.
 * @param[in, out] cells Ouput particle cell ids indexed by particle.
 * @param[in] rng_in Optional input RNG to use.
 * @param[in] attempt_max Optional maximum number of attempts to place particles
 * in each cell (default 1E8).
 */
inline void uniform_within_dmplex_cells(
    DMPlexInterfaceSharedPtr mesh, const int npart_per_cell,
    std::vector<std::vector<double>> &positions, std::vector<int> &cells,
    std::optional<std::mt19937> rng_in = std::nullopt,
    const int attempt_max = 1e8) {

  std::mt19937 rng;
  if (!rng_in) {
    rng = std::mt19937(std::random_device{}());
  } else {
    rng = rng_in.value();
  }
  const int ndim = mesh->get_ndim();
  const int cell_count = mesh->get_cell_count();
  const int npart_total = npart_per_cell * cell_count;

  // resize output space
  positions.resize(ndim);
  for (int dx = 0; dx < ndim; dx++) {
    positions[dx] = std::vector<double>(npart_total);
  }
  cells.resize(npart_total);

  // for each cell make particle positions
  int index = 0;
  std::vector<PetscScalar> proposed_position(ndim);
  for (int cx = 0; cx < cell_count; cx++) {
    auto bounding_box = mesh->dmh->get_cell_bounding_box(cx);
    std::vector<std::uniform_real_distribution<double>> dists;
    dists.reserve(ndim);
    for (int dx = 0; dx < ndim; dx++) {
      dists.push_back(std::uniform_real_distribution<double>{
          bounding_box->lower(dx), bounding_box->upper(dx)});
    }

    for (int px = 0; px < npart_per_cell; px++) {
      bool contained = false;
      int attempt_count = 0;
      while ((!contained) && (attempt_count < attempt_max)) {
        for (int dx = 0; dx < ndim; dx++) {
          proposed_position.at(dx) = dists.at(dx)(rng);
        }
        contained = mesh->dmh->cell_contains_point(cx, proposed_position);
        attempt_count++;
      }
      NESOASSERT(attempt_count < attempt_max, "Maximum attempt count reached.");
      for (int dx = 0; dx < ndim; dx++) {
        positions.at(dx).at(index) = proposed_position.at(dx);
      }
      cells.at(index) = cx;
      index++;
    }
  }
  NESOASSERT(index == npart_total, "Error creating particle positions.");
}

} // namespace NESO::Particles::PetscInterface

#endif
