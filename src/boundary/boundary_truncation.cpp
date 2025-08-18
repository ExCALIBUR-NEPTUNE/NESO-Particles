#include <neso_particles/boundary/boundary_truncation.hpp>

namespace NESO::Particles {

BoundaryTruncation::BoundaryTruncation(const int ndim,
                                       const REAL reset_distance)
    : ndim(ndim), reset_distance(reset_distance) {
  NESOASSERT((0 < ndim) && (ndim < 4), "Bad number of dimensions passed.");
}

void BoundaryTruncation::execute(
    std::shared_ptr<ParticleSubGroup> particle_sub_group,
    Sym<REAL> sym_positions, Sym<REAL> sym_time_step_proportion,
    Sym<REAL> sym_positions_previous) {

  NESOASSERT(
      contains_boundary_interaction_data(particle_sub_group, this->ndim, true),
      "This ParticleSubGroup does not have the EphemeralDats that describe "
      "the boundary interaction.");

  const REAL k_reset_distance = this->reset_distance;

  if (this->ndim == 1) {
    particle_loop(
        "BoundaryTruncation::execute_inner_1d", particle_sub_group,
        [=](auto P, auto TSP, auto PP, auto INTERSECTION_POINT,
            auto INTERSECTION_NORMAL) {
          Private::truncate_trajectory<1>(k_reset_distance, P, TSP, PP,
                                          INTERSECTION_POINT,
                                          INTERSECTION_NORMAL);
        },
        Access::write(sym_positions), Access::write(sym_time_step_proportion),
        Access::read(sym_positions_previous),
        Access::read(BoundaryInteractionSpecification::intersection_point),
        Access::read(BoundaryInteractionSpecification::intersection_normal))
        ->execute();
  } else if (this->ndim == 2) {
    particle_loop(
        "BoundaryTruncation::execute_inner_2d", particle_sub_group,
        [=](auto P, auto TSP, auto PP, auto INTERSECTION_POINT,
            auto INTERSECTION_NORMAL) {
          Private::truncate_trajectory<2>(k_reset_distance, P, TSP, PP,
                                          INTERSECTION_POINT,
                                          INTERSECTION_NORMAL);
        },
        Access::write(sym_positions), Access::write(sym_time_step_proportion),
        Access::read(sym_positions_previous),
        Access::read(BoundaryInteractionSpecification::intersection_point),
        Access::read(BoundaryInteractionSpecification::intersection_normal))
        ->execute();
  } else {
    particle_loop(
        "BoundaryTruncation::execute_inner_3d", particle_sub_group,
        [=](auto P, auto TSP, auto PP, auto INTERSECTION_POINT,
            auto INTERSECTION_NORMAL) {
          Private::truncate_trajectory<3>(k_reset_distance, P, TSP, PP,
                                          INTERSECTION_POINT,
                                          INTERSECTION_NORMAL);
        },
        Access::write(sym_positions), Access::write(sym_time_step_proportion),
        Access::read(sym_positions_previous),
        Access::read(BoundaryInteractionSpecification::intersection_point),
        Access::read(BoundaryInteractionSpecification::intersection_normal))
        ->execute();
  }
}

} // namespace NESO::Particles
