#ifndef _NESO_PARTICLES_LINEAR_INDEX_HELPER_HPP_
#define _NESO_PARTICLES_LINEAR_INDEX_HELPER_HPP_

#include "particle_group.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 * Helper struct to compute linear particle indices from an exclusive scan, cell
 * index and a layer index.
 */
struct ParticleLinearIndexDevice {
  INT *d_npart_cell_es = nullptr;

  /**
   * Get a linear index.
   *
   * @param cell Cell of particle.
   * @param layer Layer of particle.
   * @returns Linear index of particle.
   */
  inline INT get_local_linear_index(const int cell, const int layer) const {
    return this->d_npart_cell_es[cell] + layer;
  }
};

/**
 * Get a linear index helper from a ParticleGroup.
 *
 * @param particle_group ParticleGroup to get a linear index helper for.
 * @returns ParticleLinearIndexDevice for ParticleGroup.
 */
ParticleLinearIndexDevice
get_particle_linear_index_device(ParticleGroupSharedPtr particle_group);

} // namespace NESO::Particles

#endif
