#ifndef _NESO_PARTICLES_BOUNDARY_BOUNDARY_REFLECTION_HPP_
#define _NESO_PARTICLES_BOUNDARY_BOUNDARY_REFLECTION_HPP_

#include "../error_propagate.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"
#include "boundary_interaction_specification.hpp"

namespace NESO::Particles::ExternalCommon {

/**
 * Helper class to apply a reflection process to particles which intersect a
 * boundary. This implementation truncates the trajectory just before it
 * intersects the boundary whilst updating the velocity component using the
 * normal vector for the boundary.
 *
 * In addition to updating the particle positions and velocities this class
 * updates the time of the particle to the point that the particle trajectory
 * hit the wall. For each particle there should exist a Sym<REAL> property of
 * at least two components. The first component holds the proportion of the
 * current time step which has been completed. The second component holds the
 * proportion of a time step which the last update to the particle position
 * that was performed.
 *
 * For example if the users time stepping performs an time step which
 * notionally moved the particle in time by the whole time step then the first
 * and second components are both 1. If the intersection is halfway between the
 * start position and end position then on return the first and second
 * component would both be 0.5. There are two components as updating the time
 * of the particle is an iterative process which might require multiple calls
 * to execute, for example a particle bouncing in a corner.
 */
class BoundaryReflection {
protected:
  int ndim;
  REAL reset_distance;

public:
  /**
   * Create an instance to perform reflections from a boundary interaction
   * class. Note that this class does not perform calls to this boundary
   * interaction class other than those to retrieve the normal data.
   *
   * @param ndim Number of spatial dimensions.
   * @param reset_distance Optionally pass the distance from the boundary to
   * the position the particle is reset to. If this value is zero then the
   * particle will become stuck in the wall.
   */
  BoundaryReflection(const int ndim, const REAL reset_distance = 1.0e-10)
      : ndim(ndim), reset_distance(reset_distance) {}

  /**
   * Perform reflection operation for all particles passed. This method should
   * be called after the user has called @ref post_integration on the boundary
   * interaction class.
   *
   * @param particle_sub_group ParticleSubGroup of particles to
   * perform reflection operation on.
   * @param sym_positions Sym<REAL> which holds the particle positions in the
   * colllection of particles. These positions will be truncated to the
   * boundary.
   * @param sym_velocities Sym<REAL> which holds the velocities of the
   * particles. These velocities will be reflected using the normal vector of
   * the boundary.
   * @param sym_time_step_proportion Sym<REAL> of two components which holds
   * the proportion of the time step which has currently been performed in the
   * first component and in the second component the size (proportion) of the
   * last time update operation.
   * @param sym_positions_previous Sym<REAL> containing the positions at the
   * previous time step.
   */
  void execute(std::shared_ptr<ParticleSubGroup> particle_sub_group,
               Sym<REAL> sym_positions, Sym<REAL> sym_velocities,
               Sym<REAL> sym_time_step_proportion,
               Sym<REAL> sym_positions_previous);
};

} // namespace NESO::Particles::ExternalCommon

#endif
