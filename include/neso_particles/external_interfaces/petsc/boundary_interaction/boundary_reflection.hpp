#ifndef _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_REFLECTION_HPP_
#define _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_REFLECTION_HPP_

#include "boundary_interaction_2d.hpp"

namespace NESO::Particles::PetscInterface {

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
  std::shared_ptr<BoundaryInteraction2D> boundary_interaction_2d;
  int ndim;
  REAL reset_distance;
  std::shared_ptr<ErrorPropagate> ep;

  template <typename T>
  inline void execute_inner_2d(std::shared_ptr<T> particle_group,
                               Sym<REAL> sym_positions,
                               Sym<REAL> sym_velocities,
                               Sym<REAL> sym_time_step_proportion) {

    auto normal_mapper =
        this->boundary_interaction_2d->get_device_normal_mapper();
    auto k_ep = this->ep->device_ptr();
    const REAL k_reset_distance = this->reset_distance;

    particle_loop(
        "BoundaryReflection::execute_inner_2d", particle_group,
        [=](auto P, auto V, auto TSP, auto PP, auto B_P, auto B_C) {
          REAL *normal;
          const bool normal_exists = normal_mapper.get(B_C.at(2), &normal);
          NESO_KERNEL_ASSERT(normal_exists, k_ep);
          if (normal_exists) {
            // Normal vector
            const REAL n0 = normal[0];
            const REAL n1 = normal[1];
            const REAL p0 = P.at(0);
            const REAL p1 = P.at(1);
            const REAL v0 = V.at(0);
            const REAL v1 = V.at(1);
            // We don't know if the normal is inwards pointing or outwards
            // pointing.
            const REAL in_dot_product = KERNEL_DOT_PRODUCT_2D(n0, n1, v0, v1);

            // compute new velocity from reflection
            V.at(0) = v0 - 2.0 * in_dot_product * n0;
            V.at(1) = v1 - 2.0 * in_dot_product * n1;

            // Try and compute a sane new position
            // vector from intersection point back towards previous position
            const REAL oo0 = PP.at(0) - B_P.at(0);
            const REAL oo1 = PP.at(1) - B_P.at(1);
            REAL o0 = oo0;
            REAL o1 = oo1;

            const REAL o_norm2 = KERNEL_DOT_PRODUCT_2D(oo0, oo1, oo0, oo1);
            const REAL o_norm = Kernel::sqrt(o_norm2);
            const bool small_move = o_norm < (k_reset_distance * 0.1);
            const REAL o_inorm =
                small_move ? k_reset_distance : k_reset_distance / o_norm;
            o0 *= o_inorm;
            o1 *= o_inorm;
            // If the move is tiny place the particle back on the previous
            // position
            REAL np0 = small_move ? PP.at(0) : B_P.at(0) + o0;
            REAL np1 = small_move ? PP.at(1) : B_P.at(1) + o1;
            // Detect if we moved the particle back past the previous position
            // Both PP - np and PP - IP should have the same sign
            const bool moved_past_pp =
                ((PP.at(0) - np0) * o0 < 0.0) || ((PP.at(1) - np1) * o1 < 0.0);
            np0 = moved_past_pp ? PP.at(0) : np0;
            np1 = moved_past_pp ? PP.at(1) : np1;

            P.at(0) = np0;
            P.at(1) = np1;

            // Timestepping adjustment
            const REAL dist_trunc_step = o_norm2;

            const REAL f0 = p0 - PP.at(0);
            const REAL f1 = p1 - PP.at(1);
            const REAL dist_full_step = KERNEL_DOT_PRODUCT_2D(f0, f1, f0, f1);

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
          }
        },
        Access::write(sym_positions), Access::write(sym_velocities),
        Access::write(sym_time_step_proportion),
        Access::read(this->boundary_interaction_2d->previous_position_sym),
        Access::read(this->boundary_interaction_2d->boundary_position_sym),
        Access::read(this->boundary_interaction_2d->boundary_label_sym))
        ->execute();

    this->ep->check_and_throw(
        "Error executing BoundaryReflection::execute_inner_2d");
  }

public:
  /**
   * Create an instance to perform reflections from a boundary interaction
   * class. Note that this class does not perform calls to this boundary
   * interaction class other than those to retrieve the normal data.
   *
   * @param boundary_interaction_2d The boundary interaction object the user
   * used to identify the intersection of the particle trajectory and the
   * boundary.
   * @param reset_distance Optionally pass the distance from the boundary to
   * the position the particle is reset to. If this value is zero then the
   * particle will become stuck in the wall.
   */
  BoundaryReflection(
      std::shared_ptr<BoundaryInteraction2D> boundary_interaction_2d,
      const REAL reset_distance = 1.0e-10)
      : boundary_interaction_2d(boundary_interaction_2d), ndim(2),
        reset_distance(reset_distance) {
    this->ep = std::make_shared<ErrorPropagate>(
        this->boundary_interaction_2d->sycl_target);
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
  template <typename T>
  inline void execute(std::shared_ptr<T> particle_group,
                      Sym<REAL> sym_positions, Sym<REAL> sym_velocities,
                      Sym<REAL> sym_time_step_proportion) {
    NESOASSERT(this->ndim == 2, "Only implemented in 2D");
    return this->execute_inner_2d(particle_group, sym_positions, sym_velocities,
                                  sym_time_step_proportion);
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
