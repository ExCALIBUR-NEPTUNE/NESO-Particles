#ifndef _NESO_PARTICLES_EXTERNAL_INTERFACES_COMMON_BOUNDARY_REFLECTION_HPP_
#define _NESO_PARTICLES_EXTERNAL_INTERFACES_COMMON_BOUNDARY_REFLECTION_HPP_

#include "../../error_propagate.hpp"
#include "../../particle_group.hpp"
#include "../../particle_sub_group/particle_sub_group.hpp"
#include "../../boundary_interaction_specification.hpp"

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

  inline void execute_inner_2d(std::shared_ptr<ParticleSubGroup> particle_sub_group,
                               Sym<REAL> sym_positions,
                               Sym<REAL> sym_velocities,
                               Sym<REAL> sym_time_step_proportion,
                               Sym<REAL> sym_positions_previous) {

    auto particle_group = get_particle_group(particle_sub_group);
    auto sycl_target = particle_group->sycl_target;

    NESOASSERT(
        contains_boundary_interaction_data(particle_sub_group, this->ndim),
        "This ParticleSubGroup does not have the EphemeralDats that describe "
        "the boundary interaction.");

    auto ep = get_resource<ErrorPropagate, ResourceStackInterfaceErrorPropagate>(
          sycl_target->resource_stack_map, ResourceStackKeyErrorPropagate{},
          sycl_target);

    auto k_ep = ep->device_ptr();
    const REAL k_reset_distance = this->reset_distance;
    const auto k_ndim = this->ndim;

    particle_loop(
        "BoundaryReflection::execute_inner_2d", particle_sub_group,
        [=](auto P, auto V, auto TSP, auto PP, auto INTERSECTION_POINT,
            auto INTERSECTION_NORMAL) {
          REAL n[3] = {0.0, 0.0, 0.0};
          REAL p[3] = {0.0, 0.0, 0.0};
          REAL v[3] = {0.0, 0.0, 0.0};

          for(int dx=0 ; dx<k_ndim ; dx++){
            n[dx] = INTERSECTION_NORMAL.at_ephemeral(dx);
            p[dx] = P.at(dx);
            v[dx] = V.at(dx);
          }
          // We don't know if the normal is inwards pointing or outwards
          // pointing.
          const REAL in_dot_product = Kernel::dot_product_3d(n, v);

          // compute new velocity from reflection
          for(int dx=0 ; dx<k_ndim ; dx++){
            V.at(dx) = v[dx] - 2.0 * in_dot_product * n[dx];
          }

          // Try and compute a sane new position
          // vector from intersection point back towards previous position
          REAL oo[3] = {0.0, 0.0, 0.0};
          for(int dx=0 ; dx<k_ndim ; dx++){
            oo[dx] = PP.at(dx) - INTERSECTION_POINT.at_ephemeral(dx);
          }
          REAL o[3] = {0.0, 0.0, 0.0};
          for(int dx=0 ; dx<k_ndim ; dx++){
            o[dx] = oo[dx];
          }

          const REAL o_norm2 = Kernel::dot_product_3d(oo, oo);
          const REAL o_norm = Kernel::sqrt(o_norm2);
          const bool small_move = o_norm < (k_reset_distance * 0.1);
          const REAL o_inorm =
              small_move ? k_reset_distance : k_reset_distance / o_norm;
          for(int dx=0 ; dx<k_ndim ; dx++){
            o[dx] *= o_inorm;
          }

          // If the move is tiny place the particle back on the previous
          // position
          REAL np[3] = {0.0, 0.0, 0.0};
          for(int dx=0 ; dx<k_ndim ; dx++){
            np[dx] = small_move ? PP.at(dx) : INTERSECTION_POINT.at_ephemeral(dx) + o[dx];
          }
          // Detect if we moved the particle back past the previous position
          // Both PP - np and PP - IP should have the same sign
          const bool moved_past_pp = ((PP.at(0) - np[0]) * o[0] < 0.0) ||
                                     ((PP.at(1) - np[1]) * o[1] < 0.0) ||
                                     ((PP.at(2) - np[2]) * o[2] < 0.0);

          for(int dx=0 ; dx<k_ndim ; dx++){
            P.at(dx) = moved_past_pp ? PP.at(dx) : np[dx];
          }

          // Timestepping adjustment
          const REAL dist_trunc_step = o_norm;
          
          REAL f[3] = {0.0, 0.0, 0.0};
          for(int dx=0 ; dx<k_ndim ; dx++){
            f[dx] = p[dx] - PP.at(dx);
          }
          const REAL dist_full_step = Kernel::sqrt(Kernel::dot_product_3d(f, f));

          REAL tmp_prop_achieved = dist_full_step > 1.0e-16
                                       ? dist_trunc_step / dist_full_step
                                       : 1.0;
          tmp_prop_achieved =
              tmp_prop_achieved < 0.0 ? 0.0 : tmp_prop_achieved;
          tmp_prop_achieved =
              tmp_prop_achieved > 1.0 ? 1.0 : tmp_prop_achieved;

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

    ep->check_and_throw("Error executing BoundaryReflection::execute_inner_2d");
    restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyErrorPropagate{}, ep);
  }

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
  BoundaryReflection(
      const int ndim,
      const REAL reset_distance = 1.0e-10)
      : ndim(ndim),
        reset_distance(reset_distance) {
  }

  /**
   * Perform reflection operation for all particles passed. This method should
   * be called after the user has called @ref post_integration on the boundary
   * interaction class.
   *
   * @param particle_group ParticleGroup or ParticleSubGroup of particles to
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
   */
  void execute(std::shared_ptr<ParticleGroup> particle_group,
               Sym<REAL> sym_positions, Sym<REAL> sym_velocities,
               Sym<REAL> sym_time_step_proportion);

  /**
   * Perform reflection operation for all particles passed. This method should
   * be called after the user has called @ref post_integration on the boundary
   * interaction class.
   *
   * @param particle_group ParticleGroup or ParticleSubGroup of particles to
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
   */
  void execute(std::shared_ptr<ParticleSubGroup> particle_group,
               Sym<REAL> sym_positions, Sym<REAL> sym_velocities,
               Sym<REAL> sym_time_step_proportion);
};

} // namespace NESO::Particles::ExternalCommon

#endif
