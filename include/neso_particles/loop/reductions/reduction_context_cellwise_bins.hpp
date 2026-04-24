#ifndef _NESO_PARTICLES_LOOP_REDUCTIONS_REDUCTION_CONTEXT_CELLWISE_BINS_HPP_
#define _NESO_PARTICLES_LOOP_REDUCTIONS_REDUCTION_CONTEXT_CELLWISE_BINS_HPP_

#include "../../containers/index_map.hpp"
#include "../../particle_sub_group/particle_sub_group_base.hpp"
#include "../../particle_sub_group/particle_sub_group_utility.hpp"

namespace NESO::Particles {

/**
 * Type that indicates reductions should be made first by mesh cell then by bin
 * index. The intended use case is for N mesh cells and M bins per mesh cell a
 * CellDatConst exists with N cells and M rows. Particles from mesh cell c and
 * bin b will contribute to the b-th row of the matrix in mesh cell c.
 */
struct ReductionContextCellwiseBins {

  // The ParticleGroup for the particles.
  ParticleGroupSharedPtr particle_group{nullptr};

  // The ParticleSubGroup for the particles.
  ParticleSubGroupSharedPtr particle_sub_group{nullptr};

  // The partition used for describing which mesh cell and velocity bin
  // particles reside in.
  IndexMapSharedPtr<2, 1> partition{nullptr};

  /**
   * Create a reduction context for a particular partition into mesh cells and
   * bins. E.g. velocity bins.
   *
   * @param particle_group Host ParticleGroup for particles.
   * @param partition Map from cells and bins to particles.
   */
  ReductionContextCellwiseBins(ParticleGroupSharedPtr particle_group,
                               IndexMapSharedPtr<2, 1> partition)
      : particle_group(particle_group), partition(partition) {}

  /**
   * Create a reduction context for a particular partition into mesh cells and
   * bins. E.g. velocity bins.
   *
   * @param particle_group Host ParticleGroup for particles.
   * @param partition Map from cells and bins to particles.
   */
  ReductionContextCellwiseBins(ParticleSubGroupSharedPtr particle_sub_group,
                               IndexMapSharedPtr<2, 1> partition)
      : particle_group(get_particle_group(particle_sub_group)),
        particle_sub_group(particle_sub_group), partition(partition) {}

  /**
   * Releases the partition such that it can be returned to a resource stack.
   */
  inline void free() {
    this->particle_group = nullptr;
    this->particle_sub_group = nullptr;
    this->partition = nullptr;
  }
};

} // namespace NESO::Particles

#endif
