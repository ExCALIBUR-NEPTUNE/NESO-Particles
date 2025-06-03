#include <neso_particles/particle_sub_group/sub_group_selector_utility.hpp>

namespace NESO::Particles {

namespace Private {
int *get_npart_cell_device_ptr(ParticleGroupSharedPtr particle_group) {
  return particle_group->get_dat(Sym<INT>("NESO_MPI_RANK"))->d_npart_cell;
}
INT *get_npart_cell_es_device_ptr(ParticleGroupSharedPtr particle_group) {
  return particle_group->dh_npart_cell_es->d_buffer.ptr;
}
int *get_npart_cell_device_ptr(ParticleSubGroupSharedPtr particle_sub_group) {
  particle_sub_group->create_if_required();
  if (particle_sub_group->is_entire_particle_group()) {
    return get_npart_cell_device_ptr(particle_sub_group->get_particle_group());
  } else {
    const auto &selection = particle_sub_group->get_selection();
    return selection.d_npart_cell;
  }
}
INT *get_npart_cell_es_device_ptr(
    ParticleSubGroupSharedPtr particle_sub_group) {
  particle_sub_group->create_if_required();
  if (particle_sub_group->is_entire_particle_group()) {
    return get_npart_cell_es_device_ptr(
        particle_sub_group->get_particle_group());
  } else {
    const auto &selection = particle_sub_group->get_selection();
    return selection.d_npart_cell_es;
  }
}

} // namespace Private
} // namespace NESO::Particles
