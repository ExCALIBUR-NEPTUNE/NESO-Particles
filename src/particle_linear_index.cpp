#include <neso_particles/particle_linear_index.hpp>

namespace NESO::Particles {

ParticleLinearIndexDevice
get_particle_linear_index_device(ParticleGroupSharedPtr particle_group) {
  ParticleLinearIndexDevice plid;
  plid.d_npart_cell_es = particle_group->dh_npart_cell_es->d_buffer.ptr;
  return plid;
}

} // namespace NESO::Particles
