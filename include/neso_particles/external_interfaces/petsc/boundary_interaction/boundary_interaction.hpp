#ifndef _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_HPP_
#define _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_HPP_

#include "boundary_interaction_2d.hpp"
#include "boundary_interaction_3d.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Create a BoundaryInteraction3D or BoundaryInteraction2D instance and cast it
 * to a BoundaryInteractionCommon.
 *
 * @param sycl_target Compute device to use to identify intersections of
 * trajectories and the boundary.
 * @param mesh 3D or 2D DMPlex mesh interface to use.
 * @param boundary_groups Map from group IDs to the boundary labels (i.e.
 * gmsh physical lines) that form the group.
 * @param tol Tolerance for intersection of trajectories and the line
 * segments that form the boundary. If particles are passing through corners
 * try increasing this value (default 0.0).
 * @param previous_position_sym The Sym for the particle property which holds
 * the position of each particle before the positions were updated in a time
 * stepping loop. These positions are populated on call to @ref
 * pre_integration.
 */
std::shared_ptr<BoundaryInteractionCommon> create_boundary_interaction(
    SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
    std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
    const REAL tol = 0.0,
    std::optional<Sym<REAL>> previous_position_sym = std::nullopt);
} // namespace NESO::Particles::PetscInterface

#endif
