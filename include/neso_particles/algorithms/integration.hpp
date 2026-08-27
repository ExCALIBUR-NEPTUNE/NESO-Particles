#ifndef _NESO_PARTICLES_ALGORITHMS_INTEGRATION_HPP_
#define _NESO_PARTICLES_ALGORITHMS_INTEGRATION_HPP_

#include "../loop/particle_loop_functions.hpp"
#include "../loop/particle_loop_impl.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/particle_loop_sub_group_functions.hpp"
#include "../particle_sub_group/particle_sub_group_utility.hpp"

namespace NESO::Particles {

/**
 * Perform one step of Forward Euler integration.
 *
 * @param particle_sub_group Particle{Sub}Group containing particles.
 * @param position Sym of particle positions.
 * @param dt Timestep size.
 * @param velocity Sym of particle velocities.
 */
template <typename GROUP_TYPE>
inline void forward_euler(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                          Sym<REAL> position, const REAL dt,
                          Sym<REAL> velocity) {

  auto particle_group = get_particle_group(particle_sub_group);

  NESOASSERT(particle_group->contains_dat(position), "Position dat not found.");
  NESOASSERT(particle_group->contains_dat(velocity), "Velocity dat not found.");

  const int ndim = particle_group->get_dat(position)->ncomp;

  NESOASSERT(
      particle_group->get_dat(position)->ncomp == ndim,
      "Miss-match in number of components between position and velocity.");

  if (ndim == 1) {
    particle_loop(
        "Algorithms:forward_euler_1d", particle_sub_group,
        [=](auto P, auto V) { P.at(0) += dt * V.at(0); },
        Access::write(position), Access::read(velocity))
        ->execute();

  } else if (ndim == 2) {
    particle_loop(
        "Algorithms:forward_euler_2d", particle_sub_group,
        [=](auto P, auto V) {
          P.at(0) += dt * V.at(0);
          P.at(1) += dt * V.at(1);
        },
        Access::write(position), Access::read(velocity))
        ->execute();

  } else if (ndim == 3) {
    particle_loop(
        "Algorithms:forward_euler_3d", particle_sub_group,
        [=](auto P, auto V) {
          P.at(0) += dt * V.at(0);
          P.at(1) += dt * V.at(1);
          P.at(2) += dt * V.at(2);
        },
        Access::write(position), Access::read(velocity))
        ->execute();

  } else {
    particle_loop(
        "Algorithms:forward_euler_nd", particle_sub_group,
        [=](auto P, auto V) {
          for (int dx = 0; dx < ndim; dx++) {
            P.at(dx) += dt * V.at(dx);
          }
        },
        Access::write(position), Access::read(velocity))
        ->execute();
  }
}

extern template void forward_euler(ParticleGroupSharedPtr, Sym<REAL>,
                                   const REAL, Sym<REAL>);
extern template void forward_euler(ParticleSubGroupSharedPtr, Sym<REAL>,
                                   const REAL, Sym<REAL>);

} // namespace NESO::Particles

#endif
