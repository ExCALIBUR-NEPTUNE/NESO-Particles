#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/common_impl.hpp>
#include <neso_particles/external_interfaces/common/boundary_reflection.hpp>

namespace NESO::Particles::ExternalCommon {

void BoundaryReflection::execute(std::shared_ptr<ParticleGroup> particle_group,
                                 Sym<REAL> sym_positions,
                                 Sym<REAL> sym_velocities,
                                 Sym<REAL> sym_time_step_proportion) {
  NESOASSERT(this->ndim == 2, "Only implemented in 2D");
  return this->execute_inner_2d(particle_group, sym_positions, sym_velocities,
                                sym_time_step_proportion);
}

void BoundaryReflection::execute(
    std::shared_ptr<ParticleSubGroup> particle_group, Sym<REAL> sym_positions,
    Sym<REAL> sym_velocities, Sym<REAL> sym_time_step_proportion) {
  NESOASSERT(this->ndim == 2, "Only implemented in 2D");
  return this->execute_inner_2d(particle_group, sym_positions, sym_velocities,
                                sym_time_step_proportion);
}

} // namespace NESO::Particles::ExternalCommon

#endif
