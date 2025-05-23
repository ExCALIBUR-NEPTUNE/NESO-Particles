#include <neso_particles/boundary_interaction_specification.hpp>
#include <neso_particles/common_impl.hpp>

namespace NESO::Particles {

bool contains_boundary_interaction_data(
    const ParticleSubGroupSharedPtr particle_sub_group, std::optional<int> ndim,
    bool verbose) {
  bool valid = true;

  auto lambda_test_dat = [&](auto sym, auto ndim_tmp) {
    const bool contains_dat = particle_sub_group->contains_ephemeral_dat(sym);
    valid = valid && contains_dat;
    if ((ndim_tmp != std::nullopt) && contains_dat) {
      valid = valid && (particle_sub_group->get_ephemeral_dat(sym)->ncomp ==
                        ndim_tmp.value());
    }
    if (verbose && (!contains_dat)) {
      nprint("EphemeralDat:", sym.name,
             "is missing from the ParticleSubGroup.");
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

void add_boundary_interaction_ephemeral_dats(
    ParticleSubGroupSharedPtr particle_sub_group, const int ndim) {

  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_point, ndim);
  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_normal, ndim);
  particle_sub_group->add_ephemeral_dat(
      BoundaryInteractionSpecification::intersection_metadata,
      BoundaryInteractionSpecification::intersection_metadata_ncomp);

  NESOASSERT(contains_boundary_interaction_data(particle_sub_group, ndim),
             "Failed to add required boundary EphemeralDats.");
}

} // namespace NESO::Particles
