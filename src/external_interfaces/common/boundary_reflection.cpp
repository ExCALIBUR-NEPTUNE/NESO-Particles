#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/common_impl.hpp>
#include <neso_particles/external_interfaces/common/boundary_reflection.hpp>

namespace NESO::Particles::ExternalCommon {

void BoundaryReflection::execute(
    std::shared_ptr<ParticleSubGroup> particle_sub_group,
    Sym<REAL> sym_positions, Sym<REAL> sym_velocities,
    Sym<REAL> sym_time_step_proportion, Sym<REAL> sym_positions_previous) {
  NESOASSERT(this->ndim == 2, "Only implemented in 2D");

  auto particle_group = get_particle_group(particle_sub_group);
  auto sycl_target = particle_group->sycl_target;

  NESOASSERT(
      contains_boundary_interaction_data(particle_sub_group, this->ndim),
      "This ParticleSubGroup does not have the EphemeralDats that describe "
      "the boundary interaction.");

  const REAL k_reset_distance = this->reset_distance;
  const auto k_ndim = this->ndim;

  particle_loop(
      "BoundaryReflection::execute_inner_2d", particle_sub_group,
      [=](auto P, auto V, auto TSP, auto PP, auto INTERSECTION_POINT,
          auto INTERSECTION_NORMAL) {
        REAL n[3] = {0.0, 0.0, 0.0};
        REAL p[3] = {0.0, 0.0, 0.0};
        REAL v[3] = {0.0, 0.0, 0.0};

        for (int dx = 0; dx < k_ndim; dx++) {
          n[dx] = INTERSECTION_NORMAL.at_ephemeral(dx);
          p[dx] = P.at(dx);
          v[dx] = V.at(dx);
        }
        // We don't know if the normal is inwards pointing or outwards
        // pointing.
        const REAL in_dot_product = Kernel::dot_product_3d(n, v);

        // compute new velocity from reflection
        for (int dx = 0; dx < k_ndim; dx++) {
          V.at(dx) = v[dx] - 2.0 * in_dot_product * n[dx];
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

        const REAL o_norm2 = Kernel::dot_product_3d(oo, oo);
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
          np[dx] = small_move ? PP.at(dx)
                              : INTERSECTION_POINT.at_ephemeral(dx) + o[dx];
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
        const REAL dist_full_step = Kernel::sqrt(Kernel::dot_product_3d(f, f));

        REAL tmp_prop_achieved =
            dist_full_step > 1.0e-16 ? dist_trunc_step / dist_full_step : 1.0;
        tmp_prop_achieved = tmp_prop_achieved < 0.0 ? 0.0 : tmp_prop_achieved;
        tmp_prop_achieved = tmp_prop_achieved > 1.0 ? 1.0 : tmp_prop_achieved;

        // proportion along the full step that we truncated at
        const REAL proportion_achieved = Kernel::sqrt(tmp_prop_achieved);
        const REAL last_dt = TSP.at(1);
        const REAL correct_last_dt = TSP.at(1) * proportion_achieved;
        TSP.at(0) = TSP.at(0) - last_dt + correct_last_dt;
        TSP.at(1) = correct_last_dt;
      },
      Access::write(sym_positions), Access::write(sym_velocities),
      Access::write(sym_time_step_proportion),
      Access::read(sym_positions_previous),
      Access::read(BoundaryInteractionSpecification::intersection_point),
      Access::read(BoundaryInteractionSpecification::intersection_normal))
      ->execute();
}

} // namespace NESO::Particles::ExternalCommon

#endif
