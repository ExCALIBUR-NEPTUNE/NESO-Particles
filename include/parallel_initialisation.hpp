#ifndef __NESO_PARTICLES_PARALLEL_INITIALISATION
#define __NESO_PARTICLES_PARALLEL_INITIALISATION

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

  double local_point[3];
  get_point_in_local_domain(particle_group, local_point);

  auto k_P = particle_group->position_dat->cell_dat.device_ptr();
  auto k_ORIG_POS =
      (*particle_group)[Sym<REAL>("NESO_ORIG_POS")]->cell_dat.device_ptr();
  auto pl_iter_range =
      particle_group->mpi_rank_dat->get_particle_loop_iter_range();
  auto pl_stride =
      particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell =
      particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

  sycl::buffer<double, 1> b_local_point(local_point, 3);

  // store the target position on the particle and set the starting point.
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        auto a_local_point =
            b_local_point.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
          for (int nx = 0; nx < space_ncomp; nx++) {
            k_ORIG_POS[cellx][nx][layerx] = k_P[cellx][nx][layerx];
            k_P[cellx][nx][layerx] = a_local_point[nx];
          }

          NESO_PARTICLES_KERNEL_END
        });
      })
      .wait_and_throw();
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
  auto k_P = particle_group->position_dat->cell_dat.device_ptr();
  auto k_ORIG_POS =
      (*particle_group)[Sym<REAL>("NESO_ORIG_POS")]->cell_dat.device_ptr();
  auto pl_iter_range =
      particle_group->mpi_rank_dat->get_particle_loop_iter_range();
  auto pl_stride =
      particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell =
      particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  // restore the target position
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

          for (int nx = 0; nx < space_ncomp; nx++) {
            const REAL position_original = k_ORIG_POS[cellx][nx][layerx];
            const REAL position_current = k_P[cellx][nx][layerx];

            const REAL error_abs = ABS(position_original - position_current);
            const bool valid_abs = error_abs < 1.0e-6;
            const REAL position_abs = ABS(position_original);
            const REAL error_rel =
                (position_abs == 0) ? 0.0 : error_abs / position_abs;
            const bool valid_rel = error_rel < 1.0e-6;
            const bool valid = valid_abs || valid_rel;

            NESO_KERNEL_ASSERT(valid, k_ep);
            k_P[cellx][nx][layerx] = k_ORIG_POS[cellx][nx][layerx];
          }

          NESO_PARTICLES_KERNEL_END
        });
      })
      .wait_and_throw();

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
  auto k_P = particle_group->position_dat->cell_dat.device_ptr();
  auto k_ORIG_POS =
      (*particle_group)[Sym<REAL>("NESO_ORIG_POS")]->cell_dat.device_ptr();
  auto pl_iter_range =
      particle_group->mpi_rank_dat->get_particle_loop_iter_range();
  auto pl_stride =
      particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell =
      particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

  const double steps_left = ((double)num_steps) - ((double)step);
  const double inverse_steps_left = 1.0 / steps_left;

  // move each particle along the line to the destination position.
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

          for (int nx = 0; nx < space_ncomp; nx++) {
            const double offset =
                k_ORIG_POS[cellx][nx][layerx] - k_P[cellx][nx][layerx];
            k_P[cellx][nx][layerx] += inverse_steps_left * offset;
          }

          NESO_PARTICLES_KERNEL_END
        });
      })
      .wait_and_throw();
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
