#ifndef _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_UTILITY_H_
#include "../particle_sub_group/particle_sub_group.hpp"
#include "particle_loop.hpp"

namespace NESO::Particles {

/**
 * @returns The number of particles in the iteration set.
 */
inline int get_loop_npart(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {
  const int cell_start = global_info->starting_cell;
  const int cell_end = global_info->bounding_cell;

  int num_particles;
  if ((cell_end - cell_start) == 1) {
    // Single cell looping case
    // Allocate for all the particles in the cell.
    if (global_info->particle_sub_group != nullptr) {
      num_particles =
          global_info->particle_sub_group->get_npart_cell(cell_start);
    } else {
      num_particles = global_info->particle_group->get_npart_cell(cell_start);
    }
  } else {
    // Whole domain looping case
    if (global_info->particle_sub_group != nullptr) {
      num_particles = global_info->particle_sub_group->get_npart_local();
    } else {
      num_particles = global_info->particle_group->get_npart_local();
    }
  }
  return num_particles;
}

} // namespace NESO::Particles
#endif
