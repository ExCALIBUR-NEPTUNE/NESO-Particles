#ifndef _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#include "../particle_sub_group/particle_sub_group_base.hpp"
#include "particle_loop_base.hpp"

namespace NESO::Particles {

/**
 * @returns The number of particles in the iteration set.
 */
inline int get_loop_npart(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {
  const int cell_start = global_info->starting_cell;
  const int cell_end = global_info->bounding_cell;
  const bool all_cells = global_info->all_cells;

  if (all_cells) {
    // Whole domain looping case
    if (global_info->particle_sub_group != nullptr) {
      return global_info->particle_sub_group->get_npart_local();
    } else {
      return global_info->particle_group->get_npart_local();
    }
  } else {
    int num_particles = 0;
    for (int cellx = cell_start; cellx < cell_end; cellx++) {
      // Single cell looping case
      // Allocate for all the particles in the cell.
      if (global_info->particle_sub_group != nullptr) {
        num_particles += global_info->particle_sub_group->get_npart_cell(cellx);
      } else {
        num_particles += global_info->particle_group->get_npart_cell(cellx);
      }
    }
    return num_particles;
  }
}

} // namespace NESO::Particles
#endif
