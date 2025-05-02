#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_TRAJECTORY_INTERSECTION_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_TRAJECTORY_INTERSECTION_HPP_

#include "../particle_sub_group/particle_sub_group.hpp"
#include "cartesian_h_mesh.hpp"
#include "../device_functions.hpp"

namespace NESO::Particles {

class CartesianTrajectoryIntersection {
protected:

  template <typename T>
  inline void check_dat(ParticleGroupSharedPtr particle_group, Sym<T> sym,
                        const int ncomp) {
    if (!particle_group->contains_dat(sym)) {
      particle_group->add_particle_dat(sym, ncomp);
    } else {
      NESOASSERT(
          particle_group->get_dat(sym)->ncomp >= ncomp,
          "Requested dat with sym " + sym.name +
              " exists already with an insufficient number of components");
    }
  }


public:
    /// Disable (implicit) copies.
  CartesianTrajectoryIntersection(const CartesianTrajectoryIntersection &st) = delete;
  /// Disable (implicit) copies.
  CartesianTrajectoryIntersection &operator=(CartesianTrajectoryIntersection const &a) = delete;
  ~CartesianTrajectoryIntersection() = default;

  /// The Sym for the particle property which holds the position of each
  /// particle before the positions were updated in a time stepping loop. These
  /// positions are populated on call to @ref pre_integration.
  inline static const Sym<REAL> previous_position_sym = 
    Sym<REAL>("NESO_PARTICLES_CART_H_MESH_PREVIOUS_POS");
  
  /// Compute device used to find intersections.
  SYCLTargetSharedPtr sycl_target;
  /// The underlying mesh.
  CartesianHMeshSharedPtr mesh;
  /// Tolerance for intersection tests.
  REAL tolerance;

  /**
   * Most simple constructor all boundary elements are considered in boundary group 0.
   *
   * @param sycl_target Compute device to use for interactions.
   * @param mesh CartesianHMesh to detect intersections with.
   * @param tolerance Tolerance for intersection tests, default 1E-14.
   */
  CartesianTrajectoryIntersection(
    SYCLTargetSharedPtr sycl_target,
    CartesianHMeshSharedPtr mesh,
    REAL tolerance = 1.0e-14
  ) :
    sycl_target(sycl_target),
    mesh(mesh),
    tolerance(tolerance)
  {
    const int k_ndim = this->mesh->get_ndim();
    NESOASSERT(k_ndim == 2 || k_ndim == 3, 
        "This method is only implemented in 2D and 3D.");
  }

  /**
   * Prepare a ParticleGroup such that it, or sub groups based on it, can be
   * passed to pre_integration and post_integration.
   *
   * @param particle_group ParticleGroup to prepare.
   */
  inline void prepare_particle_group(ParticleGroupSharedPtr particle_group){
    this->check_dat(
      particle_group, this->previous_position_sym, this->mesh->get_ndim());
  }

  /**
   * This method should be called with a collection of particles prior to
   * updating the positions of these particles.
   *
   * @param particles ParticleGroup or ParticleSubGroup of particles whose
   * positions are about to be updated, e.g. in a time stepping operation.
   */
  template <typename T>
  inline void pre_integration(std::shared_ptr<T> particles) {
    const int k_ndim = this->mesh->get_ndim();
    particle_loop(
        "CartesianTrajectoryIntersection::pre_integration", 
        particles,
        [=](auto P, auto PP) {
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            PP.at(dimx) = P.at(dimx);
          }
        },
        Access::read(get_particle_group(particles)->position_dat->sym),
        Access::write(this->previous_position_sym))
        ->execute();
  }

   /**
   * Call after updating to find particles whose trajectories intersect the
   * CartesianHMesh boundary.
   *
   * @param particles Collection of particles, either a ParticleGroup or
   * ParticleSubGroup, to identify trajectory-boundary intersections of.
   * @returns Map from boundary groups ids, which were passed in the
   * constructor, to a ParticleSubGroup of particles which crossed the boundary
   * elements which form the boundary group.
   */
  template <typename T>
  [[nodiscard]] inline std::map<int, ParticleSubGroupSharedPtr>
  post_integration(std::shared_ptr<T> particles) {

    auto particle_group = get_particle_group(particles);

    const int k_ndim = this->mesh->get_ndim();
    NESOASSERT(k_ndim == 2 || k_ndim == 3, 
        "This method is only implemented in 2D and 3D.");

    REAL k_extents[3] = {0.0, 0.0, 0.0};
    for(int dimx=0 ; dimx<k_ndim ; dimx++){
      k_extents[dimx] = this->mesh->global_extents[dimx];
    }
    
    // Collect into a sub-group the particles which are leaving the domain.
    auto departing_particles = static_particle_sub_group(
      particles,
      [=](
        auto P
      ){
        bool outside_domain = false;
        for(int dimx=0 ; dimx<k_ndim ; dimx++){
          outside_domain = outside_domain
            || ((P.at(dimx) < 0.0) || (P.at(dimx) > k_extents[dimx]));
        }
        return outside_domain;
      },
      Access::read(particle_group->position_dat->sym)
    );

    // Create a buffer to store the information for the leaving particles
    const INT npart_leaving = departing_particles->get_npart_local();
    auto d_buffer = get_resource<BufferDevice<REAL>,
                                 ResourceStackInterfaceBufferDevice<REAL>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
        sycl_target);
    d_buffer->realloc_no_copy(k_ndim * 2 * npart_leaving);
    auto k_buffer = d_buffer->ptr;

    // Create a lookup table to convert between loop indices for the sub groups
    auto d_lut = get_resource<BufferDevice<INT>,
                                 ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    d_lut->realloc_no_copy(particle_group->get_npart_local());
    auto k_lut = d_lut->ptr;

    const REAL k_tolerance = this->tolerance;

    // Compute and store into the temporary buffer the intersection information.
    if (k_ndim == 2){
      particle_loop(
        "CartesianTrajectoryIntersection::post_integration",
        departing_particles,
        [=](
          auto INDEX,
          auto P,
          auto PP
        ){
          /**
           *       2
           *    ------ 
           *   |      |
           * 3 |      | 1
           *   |      |
           *    ------
           *       0
           */

          const INT loop_index = INDEX.get_loop_linear_index();
          k_lut[INDEX.get_local_linear_index()] = loop_index;

          REAL lambda0[4];
          REAL lambda1[4];

          const REAL pa[4][2] = {
            {PP.at(0), PP.at(1)},
            {k_extents[0], 0.0},
            {PP.at(0), PP.at(1)},
            {0.0, 0.0}
          };
          const REAL pb[4][2] = {
            {P.at(0), P.at(1)},
            {k_extents[0], k_extents[1]},
            {P.at(0), P.at(1)},
            {0.0, k_extents[1]}
          };
          const REAL px[4][2] = {
            {0.0, 0.0},
            {PP.at(0), PP.at(1)},
            {0.0, k_extents[1]},
            {PP.at(0), PP.at(1)}
          };
          const REAL py[4][2] = {
            {k_extents[0], 0.0},
            {P.at(0), P.at(1)},
            {k_extents[0], k_extents[1]},
            {P.at(0), P.at(1)}
          };

          for(int edgex=0 ; edgex<4 ; edgex++){
            line_segment_intersection_2d_lambda(
              pa[edgex][0], pa[edgex][1],
              pb[edgex][0], pb[edgex][1],
              px[edgex][0], px[edgex][1],
              py[edgex][0], py[edgex][1],
              lambda0[edgex], lambda1[edgex]);
          }

          const bool x_parallel = Kernel::abs(P.at(1) - PP.at(1)) < 1.0e-16;
          const bool y_parallel = Kernel::abs(P.at(0) - PP.at(0)) < 1.0e-16;

          lambda0[0] = x_parallel ? -10000.0 : lambda0[0];
          lambda0[1] = y_parallel ? -10000.0 : lambda0[1];
          lambda0[2] = x_parallel ? -10000.0 : lambda0[2];
          lambda0[3] = y_parallel ? -10000.0 : lambda0[3];
          lambda1[0] = x_parallel ? -10000.0 : lambda1[0];
          lambda1[1] = y_parallel ? -10000.0 : lambda1[1];
          lambda1[2] = x_parallel ? -10000.0 : lambda1[2];
          lambda1[3] = y_parallel ? -10000.0 : lambda1[3];

          bool edge_found = false;
          for(int edgex=0 ; edgex<4 ; edgex++){
           const bool in_bounds_01 =
            ((0.0 - k_tolerance) <= lambda0[edgex]) && (lambda0[edgex] <= (1.0 + k_tolerance));
          }
          

          TODO



        },
        Access::read(ParticleLoopIndex{}),
        Access::read(particle_group->position_dat->sym).
        Access::read(this->previous_position_sym)
      )->execute();
    } else {



    }



    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_lut);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<REAL>{}, d_buffer);
  }


};


}

#endif
