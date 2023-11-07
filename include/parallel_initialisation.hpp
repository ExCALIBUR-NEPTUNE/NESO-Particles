#ifndef __NESO_PARTICLES_PARALLEL_INITIALISATION
#define __NESO_PARTICLES_PARALLEL_INITIALISATION

#include "loop/particle_loop.hpp"
#include "particle_group.hpp"
#include <memory>

namespace NESO::Particles {

namespace {

/**
 * Function used by parallel_advection_initialisation to step particles along a
 * line to destination positions. Gets a safe temporary position in the local
 * subdomain from which to start particle movement.
 *
 * @param particle_group ParticleGroup being initialised.
 * @param point Output point in local subdomain.
 */
inline void get_point_in_local_domain(ParticleGroupSharedPtr particle_group,
                                      double *point) {
  particle_group->domain->mesh->get_point_in_subdomain(point);
}

/**
 *  Function used by parallel_advection_initialisation to step particles along a
 * line to destination positions. Stores the original particle positions.
 *
 *  @param particle_group ParticleGroup being initialised.
 */
inline void parallel_advection_store(ParticleGroupSharedPtr particle_group) {

  const int space_ncomp = particle_group->position_dat->ncomp;
  auto domain = particle_group->domain;
  auto sycl_target = particle_group->sycl_target;
  particle_group->add_particle_dat(ParticleDat(
      sycl_target, ParticleProp(Sym<REAL>("NESO_ORIG_POS"), space_ncomp),
      domain->mesh->get_cell_count()));

  std::vector<REAL> local_point(3);
  get_point_in_local_domain(particle_group, local_point.data());
  BufferDevice<REAL> d_local_point(sycl_target, local_point);

  const auto k_local_point = d_local_point.ptr;
  auto pos_sym = particle_group->position_dat->sym;
  ParticleLoop l(
      "parallel_advection_store", particle_group,
      [=](auto P, auto ORIG_P) {
        for (int dx = 0; dx < space_ncomp; dx++) {
          ORIG_P[dx] = P[dx];
          P[dx] = k_local_point[dx];
        }
      },
      Access::write(pos_sym), Access::write(Sym<REAL>("NESO_ORIG_POS")));
  l.execute();
}

/**
 *  Function used by parallel_advection_initialisation to step particles along a
 * line to destination positions. Sets the particle positions to the original
 * positions specified by the user.
 *
 *  @param particle_group ParticleGroup being initialised.
 */
inline void parallel_advection_restore(ParticleGroupSharedPtr particle_group) {

  auto sycl_target = particle_group->sycl_target;
  const int space_ncomp = particle_group->position_dat->ncomp;
  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  auto pos_sym = particle_group->position_dat->sym;
  ParticleLoop l(
      "parallel_advection_restore", particle_group,
      [=](auto P, auto ORIG_P) {
        for (int dx = 0; dx < space_ncomp; dx++) {
          const REAL position_original = ORIG_P[dx];
          const REAL position_current = P[dx];
          const REAL error_abs = ABS(position_original - position_current);
          const bool valid_abs = error_abs < 1.0e-6;
          const REAL position_abs = ABS(position_original);
          const REAL error_rel =
              (position_abs == 0) ? 0.0 : error_abs / position_abs;
          const bool valid_rel = error_rel < 1.0e-6;
          const bool valid = valid_abs || valid_rel;
          P[dx] = ORIG_P[dx];
          NESO_KERNEL_ASSERT(valid, k_ep);
        }
      },
      Access::write(pos_sym), Access::read(Sym<REAL>("NESO_ORIG_POS")));
  l.execute();

  ep.check_and_throw("Advected particle was very far from intended position.");

  // remove the additional ParticleDat that was added
  particle_group->remove_particle_dat(Sym<REAL>("NESO_ORIG_POS"));
}

/**
 *  Function used by parallel_advection_initialisation to step particles along a
 *  line to the original positions specified by the user.
 *
 *  @param particle_group ParticleGroup being initialised.
 *  @param num_steps Number of steps over which the stepping occurs.
 *  @param step The current step out of num_steps.
 */
inline void parallel_advection_step(ParticleGroupSharedPtr particle_group,
                                    const int num_steps, const int step) {
  auto sycl_target = particle_group->sycl_target;
  const int space_ncomp = particle_group->position_dat->ncomp;

  const double steps_left = ((double)num_steps) - ((double)step);
  const double inverse_steps_left = 1.0 / steps_left;

  auto pos_sym = particle_group->position_dat->sym;
  ParticleLoop l(
      "parallel_advection_step", particle_group,
      [=](auto P, auto ORIG_P) {
        for (int dx = 0; dx < space_ncomp; dx++) {
          const double offset = ORIG_P[dx] - P[dx];
          P[dx] += inverse_steps_left * offset;
        }
      },
      Access::write(pos_sym), Access::read(Sym<REAL>("NESO_ORIG_POS")));
  l.execute();
}

} // namespace

/**
 *  Initialisation utility to aid parallel creation of particle distributions.
 *  This function performs the all-to-all movement that occurs when a
 *  ParticleGroup contains particles with positions (far) outside the owned
 *  region of space. For example consider if each MPI rank creates N particles
 *  uniformly distributed over the entire simulation domain then the call to
 *  `hybrid_move` would have a cost equal to the number of MPI ranks squared.
 *
 *  This function gives each particle a temporary position in the owned
 *  subdomain then moves the particles in a straight line to the original
 *  positions in the positions ParticleDat. It is assumed that the simulation
 *  domain is convex.
 *
 *  This function is used by adding particles with
 *  `ParticleGroup.add_particles_local` on each rank then collectively calling
 *  this function.
 *
 *  @param particle_group ParticleGroup to initialise by moving particles to the
 * positions in the position ParticleDat.
 *  @param num_steps optional number of steps to move particles over.
 */
inline void
parallel_advection_initialisation(ParticleGroupSharedPtr particle_group,
                                  const int num_steps = 20) {

  parallel_advection_store(particle_group);
  for (int stepx = 0; stepx < num_steps; stepx++) {
    parallel_advection_step(particle_group, num_steps, stepx);
    particle_group->hybrid_move();
  }
  parallel_advection_restore(particle_group);
  particle_group->hybrid_move();
}

} // namespace NESO::Particles

#endif
