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

template void fill(ParticleGroupSharedPtr, Sym<REAL>, const REAL,
                   std::optional<int>);
template void fill(ParticleGroupSharedPtr, Sym<INT>, const INT,
                   std::optional<int>);

template void fill(ParticleSubGroupSharedPtr, Sym<REAL>, const REAL,
                   std::optional<int>);
template void fill(ParticleSubGroupSharedPtr, Sym<INT>, const INT,
                   std::optional<int>);
} // namespace NESO::Particles
