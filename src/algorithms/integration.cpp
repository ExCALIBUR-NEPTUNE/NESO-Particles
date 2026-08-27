#include <neso_particles/algorithms/integration.hpp>

namespace NESO::Particles {
template void forward_euler(ParticleGroupSharedPtr, Sym<REAL>, const REAL,
                            Sym<REAL>);
template void forward_euler(ParticleSubGroupSharedPtr, Sym<REAL>, const REAL,
                            Sym<REAL>);

} // namespace NESO::Particles
