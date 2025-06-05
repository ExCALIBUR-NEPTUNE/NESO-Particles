#include <neso_particles/loop/particle_loop_args.hpp>
#include <neso_particles/loop/particle_loop_args_impl.hpp>
#include <neso_particles/loop/particle_loop_impl.hpp>

namespace NESO::Particles {

template class ParticleLoopArgs<
    NESO::Particles::Access::Read<NESO::Particles::Sym<double>>>;

}
