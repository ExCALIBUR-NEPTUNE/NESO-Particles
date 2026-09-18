#ifndef __NESO_PARTICLES_PAIR_LOOP_PAIR_UTILITY_HPP_
#define __NESO_PARTICLES_PAIR_LOOP_PAIR_UTILITY_HPP_

#include "../containers/particle_mask.hpp"
#include "../pair_loop/particle_pair_loop_cellwise_pair_list.hpp"
#include "../particle_group.hpp"

/**
 * @defgroup particle_pair_loop_helper_functions Helper Functions
 * @ingroup particle_pair_loop
 * @details Helper functions for working with pair looping.
 */

namespace NESO::Particles {

/**
 * Set ParticleMask entries to false for particles which are referenced by a
 * pair list. Particles which are not reference do not have their masks
 * modified.
 *
 * @ingroup particle_pair_loop_helper_functions
 * @param pair_list CellwisePairList containing pairs.
 * @param particle_mask ParticleMask containing masks to set for first
 * ParticleGroup.
 */
void mask_off_referenced_particles(
    CellwisePairListAbsolute<ParticleGroup, CellwisePairList> pair_list,
    ParticleMaskSharedPtr particle_mask);

} // namespace NESO::Particles

#endif
