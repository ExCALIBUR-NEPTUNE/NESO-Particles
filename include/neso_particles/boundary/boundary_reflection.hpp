#ifndef _NESO_PARTICLES_BOUNDARY_BOUNDARY_REFLECTION_HPP_
#define _NESO_PARTICLES_BOUNDARY_BOUNDARY_REFLECTION_HPP_

#include "../error_propagate.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"
#include "boundary_interaction_specification.hpp"

namespace NESO::Particles {

namespace Private {

/**
 * Kernel that applies reflection to the particle trajectory.
 *
 * @param V Current particle position.
 * @param INTERSECTION_NORMAL EphemeralDat normal at the intersection point
 * between particle trajectory and the boundary.
 */
template <int k_ndim>
inline void
reflect_trajectory(Access::ParticleDat::Write<REAL> V,
                   Access::ParticleDat::Read<REAL> INTERSECTION_NORMAL) {
  REAL n[3] = {0.0, 0.0, 0.0};
  REAL v[3] = {0.0, 0.0, 0.0};

  for (int dx = 0; dx < k_ndim; dx++) {
    n[dx] = INTERSECTION_NORMAL.at_ephemeral(dx);
    v[dx] = V.at(dx);
  }
  // We don't know if the normal is inwards pointing or outwards
  // pointing.
  const REAL in_dot_product = Kernel::dot_product<k_ndim>(n, v);

  // compute new velocity from reflection
  for (int dx = 0; dx < k_ndim; dx++) {
    V.at(dx) = v[dx] - 2.0 * in_dot_product * n[dx];
  }
}
} // namespace Private

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
 *
 * This class assumes that the boundary interaction data exists in the
 * standardised EphemeralDats that describe particle-boundary intersections. See
 * the BoundaryInteractionSpecification type for more details.
 */
class BoundaryReflection {
protected:
  int ndim;
  REAL reset_distance;

public:
  /**
   * Create an instance to perform reflections from a boundary interaction
   * class.
   *
   * @param ndim Number of spatial dimensions.
   * @param reset_distance Optionally pass the distance from the boundary to
   * the position the particle is reset to. If this value is zero then the
   * particle will become stuck in the wall.
   */
  BoundaryReflection(const int ndim, const REAL reset_distance = 1.0e-10);

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

} // namespace NESO::Particles

#endif
