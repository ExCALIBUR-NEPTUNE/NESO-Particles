#include <neso_particles/utility_particle_io.hpp>

namespace NESO::Particles {

template void get_vtk_trajectory_line<ParticleGroup>(
    std::shared_ptr<ParticleGroup> particle_sub_group, Sym<REAL> sym_start,
    Sym<REAL> sym_end, std::vector<VTK::UnstructuredCell> &trajectory);
template void get_vtk_trajectory_line<ParticleSubGroup>(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<REAL> sym_start,
    Sym<REAL> sym_end, std::vector<VTK::UnstructuredCell> &trajectory);

} // namespace NESO::Particles
