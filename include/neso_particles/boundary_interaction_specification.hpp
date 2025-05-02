#ifndef _NESO_PARTICLES_BOUNDARY_INTERACTION_SPECIFICATION_HPP_
#define _NESO_PARTICLES_BOUNDARY_INTERACTION_SPECIFICATION_HPP_

#include "particle_spec.hpp"
#include "particle_sub_group/particle_sub_group_base.hpp"
#include "typedefs.hpp"
#include <optional>

namespace NESO::Particles {

/**
 * This class defines the EphemeralDats which describe the intersection between
 * particle trajectories and the mesh boundary.
 */
struct BoundaryInteractionSpecification {
  /// The Sym corresonding to the intersection point.
  const inline static Sym<REAL> intersection_point =
      Sym<REAL>("NESO_PARTICLES_BOUNDARY_INTERSECTION_POINT");
  /// The Sym corresonding to the normal vector at the intersection point.
  const inline static Sym<REAL> intersection_normal =
      Sym<REAL>("NESO_PARTICLES_BOUNDARY_NORMAL");
  /// The Sym corresonding to the boundary group and geometry ID.
  const inline static Sym<INT> intersection_metadata =
      Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA");
  /// The number of components that the intersection_metadata EphemeralDat
  /// should have.
  const inline static int intersection_metadata_ncomp = 2;
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
inline bool contains_boundary_interaction_data(
    const ParticleSubGroupSharedPtr particle_sub_group,
    std::optional<int> ndim = std::nullopt) {
  bool valid = true;

  auto lambda_test_dat = [&](auto sym, auto ndim_tmp) {
    const bool contains_dat = particle_sub_group->contains_ephemeral_dat(sym);
    valid = valid && contains_dat;
    if ((ndim_tmp != std::nullopt) && contains_dat) {
      valid = valid && (particle_sub_group->get_ephemeral_dat(sym)->ncomp ==
                        ndim_tmp.value());
    }
  };

  lambda_test_dat(BoundaryInteractionSpecification::intersection_point, ndim);
  lambda_test_dat(BoundaryInteractionSpecification::intersection_normal, ndim);
  lambda_test_dat(
      BoundaryInteractionSpecification::intersection_metadata,
      std::optional<int>(
          BoundaryInteractionSpecification::intersection_metadata_ncomp));

  return valid;
}

} // namespace NESO::Particles

#endif
