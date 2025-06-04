#ifndef _NESO_PARTICLES_BOUNDARY_BOUNDARY_INTERACTION_SPECIFICATION_HPP_
#define _NESO_PARTICLES_BOUNDARY_BOUNDARY_INTERACTION_SPECIFICATION_HPP_

#include "../particle_spec.hpp"
#include "../particle_sub_group/particle_sub_group_base.hpp"
#include "../typedefs.hpp"
#include <optional>

namespace NESO::Particles {

/**
 * This class defines the EphemeralDats which describe the intersection between
 * particle trajectories and the mesh boundary.
 */
struct BoundaryInteractionSpecification {
  /// The Sym corresonding to the intersection point.
  static const inline Sym<REAL> intersection_point =
      Sym<REAL>("NESO_PARTICLES_BOUNDARY_INTERSECTION_POINT");
  /// The Sym corresonding to the normal vector at the intersection point.
  static const inline Sym<REAL> intersection_normal =
      Sym<REAL>("NESO_PARTICLES_BOUNDARY_NORMAL");
  /// The Sym corresonding to the boundary group and geometry ID.
  static const inline Sym<INT> intersection_metadata =
      Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA");
  /// The number of components that the intersection_metadata EphemeralDat
  /// should have.
  static constexpr inline int intersection_metadata_ncomp = 2;
};

/**
 * Test if a ParticleSubGroup currently contains the EphemeralDats that describe
 * trajectory-boundary intersections.
 *
 * @param particle_sub_group ParticleSubGroup to test for boundary intersection
 * data.
 * @param ndim Optionally specify the expected number of spatial dimensions.
 * @returns True if the particle sub group contains all of the required
 * EphemeralDats to describe intersections between particle trajectories and the
 * mesh boundary.
 */
bool contains_boundary_interaction_data(
    const ParticleSubGroupSharedPtr particle_sub_group,
    std::optional<int> ndim = std::nullopt, bool verbose = false);

/**
 * Add the required EphemeralDats to a ParticleSubGroup that are required by the
 * standardised boundary interaction specification.
 *
 * @param particle_sub_group ParticleSubGroup to add EphemeralDats to.
 * @param ndim Number of spatial dimensions.
 */
void add_boundary_interaction_ephemeral_dats(
    ParticleSubGroupSharedPtr particle_sub_group, const int ndim);

} // namespace NESO::Particles

#endif
