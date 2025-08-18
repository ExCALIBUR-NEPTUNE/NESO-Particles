#ifndef _NESO_PARTICLES_BOUNDARY_BOUNDARY_TRUNCATION_HPP_
#define _NESO_PARTICLES_BOUNDARY_BOUNDARY_TRUNCATION_HPP_

#include "../error_propagate.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"
#include "boundary_interaction_specification.hpp"

namespace NESO::Particles {

namespace Private {

/**
 * Kernel that applies trunctation to the particle trajectory.
 *
 * @param P Current particle position.
 * @param TSP Time step proportion.
 * @param PP Previous particle position.
 * @param INTERSECTION_POINT EphemeralDat intersection point between particle
 * trajectory and the boundary.
 * @param INTERSECTION_NORMAL EphemeralDat normal at the intersection point
 * between particle trajectory and the boundary.
 */
template <int k_ndim>
inline void truncate_trajectory(
    const REAL k_reset_distance, Access::ParticleDat::Write<REAL> P,
    Access::ParticleDat::Write<REAL> TSP, Access::ParticleDat::Read<REAL> PP,
    Access::ParticleDat::Read<REAL> INTERSECTION_POINT,
    [[maybe_unused]] Access::ParticleDat::Read<REAL> INTERSECTION_NORMAL) {
  REAL p[3] = {0.0, 0.0, 0.0};

  for (int dx = 0; dx < k_ndim; dx++) {
    p[dx] = P.at(dx);
  }

  // Try and compute a sane new position
  // vector from intersection point back towards previous position
  REAL oo[3] = {0.0, 0.0, 0.0};
  for (int dx = 0; dx < k_ndim; dx++) {
    oo[dx] = PP.at(dx) - INTERSECTION_POINT.at_ephemeral(dx);
  }
  REAL o[3] = {0.0, 0.0, 0.0};
  for (int dx = 0; dx < k_ndim; dx++) {
    o[dx] = oo[dx];
  }

  const REAL o_norm2 = Kernel::dot_product<k_ndim>(oo, oo);
  const REAL o_norm = Kernel::sqrt(o_norm2);
  const bool small_move = o_norm < (k_reset_distance * 0.1);
  const REAL o_inorm =
      small_move ? k_reset_distance : k_reset_distance / o_norm;
  for (int dx = 0; dx < k_ndim; dx++) {
    o[dx] *= o_inorm;
  }

  // If the move is tiny place the particle back on the previous
  // position
  REAL np[3] = {0.0, 0.0, 0.0};
  for (int dx = 0; dx < k_ndim; dx++) {
    np[dx] =
        small_move ? PP.at(dx) : INTERSECTION_POINT.at_ephemeral(dx) + o[dx];
  }

  // Detect if we moved the particle back past the previous position
  // Both PP - np and PP - IP should have the same sign
  bool moved_past_pp = false;
  for (int dx = 0; dx < k_ndim; dx++) {
    moved_past_pp = moved_past_pp || ((PP.at(dx) - np[dx]) * o[dx] < 0.0);
  }

  for (int dx = 0; dx < k_ndim; dx++) {
    P.at(dx) = moved_past_pp ? PP.at(dx) : np[dx];
  }

  // Timestepping adjustment
  const REAL dist_trunc_step = o_norm;

  REAL f[3] = {0.0, 0.0, 0.0};
  for (int dx = 0; dx < k_ndim; dx++) {
    f[dx] = p[dx] - PP.at(dx);
  }
  const REAL dist_full_step = Kernel::sqrt(Kernel::dot_product<k_ndim>(f, f));

  REAL tmp_prop_achieved =
      dist_full_step > 1.0e-16 ? dist_trunc_step / dist_full_step : 1.0;
  tmp_prop_achieved = tmp_prop_achieved < 0.0 ? 0.0 : tmp_prop_achieved;
  tmp_prop_achieved = tmp_prop_achieved > 1.0 ? 1.0 : tmp_prop_achieved;

  // proportion along the full step that we truncated at
  const REAL proportion_achieved = tmp_prop_achieved;
  const REAL last_dt = TSP.at(1);
  const REAL correct_last_dt = TSP.at(1) * proportion_achieved;
  TSP.at(0) = TSP.at(0) - last_dt + correct_last_dt;
  TSP.at(1) = correct_last_dt;
}

} // namespace Private

/**
 * Helper class to apply a truncation process to particles which intersect a
 * boundary. This implementation truncates the trajectory just before it
 * intersects the boundary and updates the time of the individual particle to
 * the predicted intersection time with the boundary.
 *
 * In addition to updating the particle positions and velocities this class
 * updates the time of the particle to the point that the particle trajectory
 * hit the wall. For each particle there should exist a Sym<REAL> property of
 * at least two components. The first component holds the proportion of the
 * current time step which has been completed. The second component holds the
 * proportion of a time step which the last update to the particle position
 * that was performed. This time reversal step assumes a Forward Euler time
 * integration scheme was used for the whole time step.
 *
 * For example if the users time stepping performs an time step which
 * notionally moved the particle in time by the whole time step then the first
 * and second components are both 1. If the intersection is halfway between the
 * start position and end position then on return the first and second
 * component would both be 0.5. There are two components as updating the time
 * of the particle is an iterative process which might require multiple calls
 * to execute, for example a particle bouncing in a corner.
 *
 * This class assumes that the boundary interaction data exists in the
 * standardised EphemeralDats that describe particle-boundary intersections. See
 * the BoundaryInteractionSpecification type for more details.
 */
class BoundaryTruncation {
protected:
  int ndim;
  REAL reset_distance;

public:
  /**
   * Create an instance to perform truncations from a boundary interaction
   * class. Note that this class does not perform calls to this boundary
   * interaction class other than those to retrieve the normal data.
   *
   * @param ndim Number of spatial dimensions.
   * @param reset_distance Optionally pass the distance from the boundary to
   * the position the particle is reset to. If this value is zero then the
   * particle will become stuck in the wall.
   */
  BoundaryTruncation(const int ndim, const REAL reset_distance = 1.0e-10);

  /**
   * Perform a truncation operation for all particles passed. This method should
   * be called after the user has called @ref post_integration on the boundary
   * interaction class.
   *
   * @param particle_sub_group ParticleSubGroup of particles to
   * perform reflection operation on.
   * @param sym_positions Sym<REAL> which holds the particle positions in the
   * colllection of particles. These positions will be truncated to the
   * boundary.
   * @param sym_time_step_proportion Sym<REAL> of two components which holds
   * the proportion of the time step which has currently been performed in the
   * first component and in the second component the size (proportion) of the
   * last time update operation.
   * @param sym_positions_previous Sym<REAL> containing the positions at the
   * previous time step.
   */
  void execute(std::shared_ptr<ParticleSubGroup> particle_sub_group,
               Sym<REAL> sym_positions, Sym<REAL> sym_time_step_proportion,
               Sym<REAL> sym_positions_previous);
};

} // namespace NESO::Particles

#endif
