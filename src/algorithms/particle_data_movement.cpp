#include <neso_particles/algorithms/particle_data_movement.hpp>

namespace NESO::Particles {

template void
copy_ephemeral_dat_to_particle_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<INT> sym_src, Sym<INT> sym_dst);
template void
copy_ephemeral_dat_to_particle_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<REAL> sym_src, Sym<REAL> sym_dst);

template void
copy_particle_dat_to_ephemeral_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<INT> sym_src, Sym<INT> sym_dst);
template void
copy_particle_dat_to_ephemeral_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<REAL> sym_src, Sym<REAL> sym_dst);

} // namespace NESO::Particles
