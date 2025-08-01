#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_TRAJECTORY_INTERSECTION_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_TRAJECTORY_INTERSECTION_HPP_

#include "../algorithms/unseen_value_extractor.hpp"
#include "../boundary/boundary_interaction_specification.hpp"
#include "../boundary/boundary_mesh_interface.hpp"
#include "../device_functions.hpp"
#include "../particle_group_impl.hpp"
#include "../particle_sub_group/particle_loop_sub_group_functions.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"
#include "cartesian_h_mesh.hpp"
#include "cartesian_h_mesh_function.hpp"
#include <array>
#include <map>

namespace NESO::Particles {

/**
 * This type implements trajectory intersection detection between particles and
 * a CartesianHMesh.
 */
class CartesianTrajectoryIntersection {
protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  std::array<INT, 6> element_offsets;
  std::array<INT, 6> element_strides0;
  std::array<INT, 6> element_strides1;
  REAL inverse_cell_width_fine;

  std::map<int, std::shared_ptr<BoundaryMeshInterface>>
      map_groups_boundary_interface;
  std::map<int, std::shared_ptr<UnseenValueExtractor>>
      map_groups_unseen_value_extractor;

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

  void setup();

  template <typename T>
  inline void pre_integration_inner(std::shared_ptr<T> particles) {
    NESOASSERT(get_particle_group(particles)->contains_dat(
                   this->previous_position_sym, this->mesh->get_ndim()),
               "Could not find dat for previous positions. Was "
               "prepare_particle_group called?");

    const int k_ndim = this->mesh->get_ndim();
    particle_loop(
        "CartesianTrajectoryIntersection::pre_integration", particles,
        [=](auto P, auto PP) {
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            PP.at(dimx) = P.at(dimx);
          }
        },
        Access::read(get_particle_group(particles)->position_dat->sym),
        Access::write(this->previous_position_sym))
        ->execute();
  }

  template <typename T>
  [[nodiscard]] inline std::map<int, ParticleSubGroupSharedPtr>
  post_integration_inner(std::shared_ptr<T> particles) {

    auto particle_group = get_particle_group(particles);

    const int k_ndim = this->mesh->get_ndim();
    NESOASSERT(k_ndim == 2 || k_ndim == 3,
               "This method is only implemented in 2D and 3D.");

    std::array<REAL, 3> k_extents = {0.0, 0.0, 0.0};
    for (int dimx = 0; dimx < k_ndim; dimx++) {
      k_extents[dimx] = this->mesh->global_extents[dimx];
    }

    // Collect into a sub-group the particles which are leaving the domain.
    auto departing_particles = static_particle_sub_group(
        particles,
        [=](auto P) {
          bool outside_domain = false;
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            outside_domain = outside_domain || ((P.at(dimx) < 0.0) ||
                                                (P.at(dimx) > k_extents[dimx]));
          }
          return outside_domain;
        },
        Access::read(particle_group->position_dat->sym));

    // Create a buffer to store the information for the leaving particles
    const INT npart_leaving = departing_particles->get_npart_local();
    auto d_buffer = get_resource<BufferDevice<REAL>,
                                 ResourceStackInterfaceBufferDevice<REAL>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
        sycl_target);
    d_buffer->realloc_no_copy(k_ndim * npart_leaving);
    auto k_buffer = d_buffer->ptr;

    auto d_buffer_int = get_resource<BufferDevice<INT>,
                                     ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    d_buffer_int->realloc_no_copy(npart_leaving);
    auto k_buffer_int = d_buffer_int->ptr;

    // Create a lookup table to convert between loop indices for the sub groups
    auto d_lut = get_resource<BufferDevice<INT>,
                              ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    d_lut->realloc_no_copy(particle_group->get_npart_local());
    auto k_lut = d_lut->ptr;

    const REAL k_tolerance = this->tolerance;

    // Compute and store into the temporary buffer the intersection information.
    if (k_ndim == 2) {
      particle_loop(
          "CartesianTrajectoryIntersection::post_integration",
          departing_particles,
          [=](auto INDEX, auto P, auto PP) {
            /**
             *       2
             *    ------
             *   |      |
             * 3 |      | 1
             *   |      |
             *    ------
             *       0
             */

            REAL xi0, xi1;
            REAL yi0, yi1;
            INT edges0, edges1;
            bool exists0, exists1;
            {
              // line is in x
              // x -> x
              // y -> y
              const bool plane_base = P.at(1) - k_tolerance <= 0.0;
              edges0 = plane_base ? 0 : 2;
              const REAL ax = PP.at(0);
              const REAL ay = PP.at(1);
              const REAL bx = P.at(0);
              const REAL by = P.at(1);
              const REAL p0x = 0.0;
              const REAL p0y = plane_base ? 0.0 : k_extents[1];
              const REAL p1x = k_extents[0];

              exists0 = line_segment_intersection_2d_x_axis_aligned(
                  ax, ay, bx, by, p0x, p0y, p1x, xi0, yi0, k_tolerance);
            }
            {
              // line is in y
              // x -> y
              // y -> x
              const bool plane_base = P.at(0) - k_tolerance <= 0.0;
              edges1 = plane_base ? 3 : 1;
              const REAL ax = PP.at(1);
              const REAL ay = PP.at(0);
              const REAL bx = P.at(1);
              const REAL by = P.at(0);
              const REAL p0x = 0.0;
              const REAL p0y = plane_base ? 0.0 : k_extents[0];
              const REAL p1x = k_extents[1];

              exists1 = line_segment_intersection_2d_x_axis_aligned(
                  ax, ay, bx, by, p0x, p0y, p1x, yi1, xi1, k_tolerance);
            }

            const REAL xi_write = exists0 ? xi0 : xi1;
            const REAL yi_write = exists0 ? yi0 : yi1;
            const INT edge_write = exists0 ? edges0 : edges1;
            const bool found = exists0 || exists1;

            const INT loop_index = INDEX.get_loop_linear_index();
            k_buffer[npart_leaving * 0 + loop_index] = xi_write;
            k_buffer[npart_leaving * 1 + loop_index] = yi_write;
            k_buffer_int[loop_index] = edge_write;
            k_lut[INDEX.get_local_linear_index()] = found ? loop_index : -1;
          },
          Access::read(ParticleLoopIndex{}),
          Access::read(particle_group->position_dat->sym),
          Access::read(this->previous_position_sym))
          ->execute();
    } else {
      particle_loop(
          "CartesianTrajectoryIntersection::post_integration",
          departing_particles,
          [=](auto INDEX, auto P, auto PP) {
            /**
             *       2
             *    ------    y
             *   |      |   ^     Bottom: 4
             * 3 |      | 1 |     Top:    5
             *   |      |   -> x
             *    ------
             *       0
             */

            REAL xi[3];
            REAL yi[3];
            REAL zi[3];
            bool exists[3];
            INT faces[3];

            {
              // xy plane is xy
              // x -> x
              // y -> y
              // z -> z
              const bool plane_base = P.at(2) - k_tolerance <= 0.0;
              faces[0] = plane_base ? 4 : 5;
              const REAL ax = PP.at(0);
              const REAL ay = PP.at(1);
              const REAL az = PP.at(2);
              const REAL bx = P.at(0);
              const REAL by = P.at(1);
              const REAL bz = P.at(2);
              const REAL p0x = 0.0;
              const REAL p0y = 0.0;
              const REAL p0z = plane_base ? 0.0 : k_extents[2];
              const REAL p1x = k_extents[0];
              const REAL p2y = k_extents[1];

              exists[0] = plane_intersection_3d_xy_plane_aligned(
                  ax, ay, az, bx, by, bz, p0x, p0y, p0z, p1x, p2y, xi[0], yi[0],
                  zi[0], k_tolerance);
            }

            {
              // xy plane is xz
              // x -> x
              // y -> z
              // z -> y
              const bool plane_base = P.at(1) - k_tolerance <= 0.0;
              faces[1] = plane_base ? 0 : 2;
              const REAL ax = PP.at(0);
              const REAL ay = PP.at(2);
              const REAL az = PP.at(1);
              const REAL bx = P.at(0);
              const REAL by = P.at(2);
              const REAL bz = P.at(1);
              const REAL p0x = 0.0;
              const REAL p0y = 0.0;
              const REAL p0z = plane_base ? 0.0 : k_extents[1];
              const REAL p1x = k_extents[0];
              const REAL p2y = k_extents[2];

              exists[1] = plane_intersection_3d_xy_plane_aligned(
                  ax, ay, az, bx, by, bz, p0x, p0y, p0z, p1x, p2y, xi[1], zi[1],
                  yi[1], k_tolerance);
            }

            {
              // xy plane is yz
              // x -> y
              // y -> z
              // z -> x
              const bool plane_base = P.at(0) - k_tolerance <= 0.0;
              faces[2] = plane_base ? 3 : 1;
              const REAL ax = PP.at(1);
              const REAL ay = PP.at(2);
              const REAL az = PP.at(0);
              const REAL bx = P.at(1);
              const REAL by = P.at(2);
              const REAL bz = P.at(0);
              const REAL p0x = 0.0;
              const REAL p0y = 0.0;
              const REAL p0z = plane_base ? 0.0 : k_extents[0];
              const REAL p1x = k_extents[1];
              const REAL p2y = k_extents[2];

              exists[2] = plane_intersection_3d_xy_plane_aligned(
                  ax, ay, az, bx, by, bz, p0x, p0y, p0z, p1x, p2y, yi[2], zi[2],
                  xi[2], k_tolerance);
            }

            bool found = false;
            REAL xi_write = 0.0;
            REAL yi_write = 0.0;
            REAL zi_write = 0.0;
            INT face_write = -1;
            for (int facex = 0; facex < 3; facex++) {
              const bool exists_inner = exists[facex];
              xi_write = exists_inner ? xi[facex] : xi_write;
              yi_write = exists_inner ? yi[facex] : yi_write;
              zi_write = exists_inner ? zi[facex] : zi_write;

              face_write = exists_inner ? faces[facex] : face_write;
              found = found || exists_inner;
            }

            const INT loop_index = INDEX.get_loop_linear_index();
            k_buffer[npart_leaving * 0 + loop_index] = xi_write;
            k_buffer[npart_leaving * 1 + loop_index] = yi_write;
            k_buffer[npart_leaving * 2 + loop_index] = zi_write;
            k_buffer_int[loop_index] = face_write;

            k_lut[INDEX.get_local_linear_index()] = found ? loop_index : -1;
          },
          Access::read(ParticleLoopIndex{}),
          Access::read(particle_group->position_dat->sym),
          Access::read(this->previous_position_sym))
          ->execute();
    }

    ErrorPropagate ep(this->sycl_target);
    auto k_ep = ep.device_ptr();
    particle_loop(
        "CartesianTrajectoryIntersection::check", departing_particles,
        [=](auto INDEX) {
          NESO_KERNEL_ASSERT(k_lut[INDEX.get_local_linear_index()] > -1, k_ep);
        },
        Access::read(ParticleLoopIndex{}))
        ->execute();
    ep.check_and_throw(
        "Failed to find boundary information for departing particle.");

    std::array<INT, 6> k_boundary_group_map;
    const INT max_edge = (k_ndim == 2) ? 4 : 6;
    for (const auto &bx : this->boundary_groups) {
      const int group = bx.first;
      for (const int ex : bx.second) {
        NESOASSERT((-1 < ex) && (ex < max_edge),
                   "Bad entry in boundary group map.");
        k_boundary_group_map[ex] = group;
      }
    }

    // Can be described in terms of EphemeralDats if EphemeralDats can be used
    // to make ParticleSubGroups or a Partition?
    std::map<int, ParticleSubGroupSharedPtr> return_map;
    for (const auto &group : this->boundary_groups) {
      const int k_group = group.first;
      return_map[k_group] = static_particle_sub_group(
          departing_particles,
          [=](auto INDEX) {
            const INT index = k_lut[INDEX.get_local_linear_index()];
            const INT facet_id = k_buffer_int[index];
            return k_boundary_group_map[facet_id] == k_group;
          },
          Access::read(ParticleLoopIndex{}));
      add_boundary_interaction_ephemeral_dats(return_map[k_group], k_ndim);
    }

    auto k_element_strides0 = this->element_strides0;
    auto k_element_strides1 = this->element_strides1;
    auto k_element_offsets = this->element_offsets;
    auto k_inverse_cell_width_fine = this->inverse_cell_width_fine;

    for (const auto &group : return_map) {
      particle_loop(
          "CartesianTrajectoryIntersection::collect_information", group.second,
          [=](auto INDEX, auto INTERSECTION_POINT, auto INTERSECTION_NORMAL,
              auto INTERSECTION_METADATA) {
            const INT index = k_lut[INDEX.get_local_linear_index()];
            const INT facet_id = k_buffer_int[index];
            INTERSECTION_METADATA.at_ephemeral(0) =
                k_boundary_group_map[facet_id];

            constexpr REAL normals[6][3] = {{0.0, 1.0, 0.0},  {-1.0, 0.0, 0.0},
                                            {0.0, -1.0, 0.0}, {1.0, 0.0, 0.0},
                                            {0.0, 0.0, 1.0},  {0.0, 0.0, -1.0}};

            for (int dx = 0; dx < k_ndim; dx++) {
              INTERSECTION_POINT.at_ephemeral(dx) =
                  k_buffer[npart_leaving * dx + index];
              INTERSECTION_NORMAL.at_ephemeral(dx) = normals[facet_id][dx];
            }

            constexpr INT facet_to_indices[6][2] = {{0, 2}, {1, 2}, {0, 2},
                                                    {1, 2}, {0, 1}, {0, 1}};

            REAL facet_point[2] = {0.0, 0.0};
            facet_point[0] =
                INTERSECTION_POINT.at_ephemeral(facet_to_indices[facet_id][0]);
            facet_point[1] = (k_ndim == 3) ? INTERSECTION_POINT.at_ephemeral(
                                                 facet_to_indices[facet_id][1])
                                           : 0.0;

            INT facet_cell[2] = {0, 0};
            facet_cell[0] = Kernel::min(
                k_element_strides0[facet_id] - 1,
                static_cast<INT>(k_inverse_cell_width_fine * facet_point[0]));
            facet_cell[1] = Kernel::min(
                k_element_strides1[facet_id] - 1,
                static_cast<INT>(k_inverse_cell_width_fine * facet_point[1]));

            const INT element_id = k_element_offsets[facet_id] + facet_cell[0] +
                                   facet_cell[1] * k_element_strides0[facet_id];

            INTERSECTION_METADATA.at_ephemeral(1) = element_id;
          },
          Access::read(ParticleLoopIndex{}),
          Access::write(BoundaryInteractionSpecification::intersection_point),
          Access::write(BoundaryInteractionSpecification::intersection_normal),
          Access::write(
              BoundaryInteractionSpecification::intersection_metadata))
          ->execute();

      auto new_geoms =
          this->map_groups_unseen_value_extractor.at(group.first)
              ->extract(group.second,
                        BoundaryInteractionSpecification::intersection_metadata,
                        1, true);
      std::vector<std::pair<int, int>> new_potentialy_hit_geoms;
      new_potentialy_hit_geoms.reserve(new_geoms.size());
      for (auto &geomx : new_geoms) {
        const int owning_rank = this->mesh->get_face_id_owning_rank(geomx);
        new_potentialy_hit_geoms.push_back({owning_rank, geomx});
      }
      this->map_groups_boundary_interface.at(group.first)
          ->extend_exchange_pattern(new_potentialy_hit_geoms);
    }

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_lut);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<REAL>{}, d_buffer);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_buffer_int);
    return return_map;
  }

public:
  /// Disable (implicit) copies.
  CartesianTrajectoryIntersection(const CartesianTrajectoryIntersection &st) =
      delete;
  /// Disable (implicit) copies.
  CartesianTrajectoryIntersection &
  operator=(CartesianTrajectoryIntersection const &a) = delete;
  ~CartesianTrajectoryIntersection();

  /// The Sym for the particle property which holds the position of each
  /// particle before the positions were updated in a time stepping loop. These
  /// positions are populated on call to @ref pre_integration.
  inline static const Sym<REAL> previous_position_sym =
      Sym<REAL>("NESO_PARTICLES_CART_H_MESH_PREVIOUS_POS");

  /// Compute device used to find intersections.
  SYCLTargetSharedPtr sycl_target;
  /// The underlying mesh.
  CartesianHMeshSharedPtr mesh;
  /// Map from boundary group to faces/edges that form the group.
  std::map<int, std::vector<int>> boundary_groups;
  /// Tolerance for intersection tests.
  REAL tolerance;

  /**
   * @param sycl_target Compute device to use for interactions.
   * @param mesh CartesianHMesh to detect intersections with.
   * @param boundary_groups Map from boundary group to edges/faces indices that
   * form the group.
   * @param tolerance Tolerance for intersection tests, default 1E-14.
   */
  CartesianTrajectoryIntersection(
      SYCLTargetSharedPtr sycl_target, CartesianHMeshSharedPtr mesh,
      std::map<int, std::vector<int>> boundary_groups,
      REAL tolerance = 1.0e-14);

  /**
   * Prepare a ParticleGroup such that it, or sub groups based on it, can be
   * passed to pre_integration and post_integration.
   *
   * @param particle_group ParticleGroup to prepare.
   */
  void prepare_particle_group(ParticleGroupSharedPtr particle_group);

  /**
   * This method should be called with a collection of particles prior to
   * updating the positions of these particles.
   *
   * @param particles ParticleGroup or ParticleSubGroup of particles whose
   * positions are about to be updated, e.g. in a time stepping operation.
   */
  void pre_integration(std::shared_ptr<ParticleGroup> particles);

  /**
   * This method should be called with a collection of particles prior to
   * updating the positions of these particles.
   *
   * @param particles ParticleGroup or ParticleSubGroup of particles whose
   * positions are about to be updated, e.g. in a time stepping operation.
   */
  void pre_integration(std::shared_ptr<ParticleSubGroup> particles);

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
  [[nodiscard]] std::map<int, ParticleSubGroupSharedPtr>
  post_integration(std::shared_ptr<ParticleGroup> particles);

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
  [[nodiscard]] std::map<int, ParticleSubGroupSharedPtr>
  post_integration(std::shared_ptr<ParticleSubGroup> particles);

  /**
   * Explicitly free the resources, e.g. MPI communicators without relying on
   * collective destructor calls. Should be called collectively on the
   * communicator.
   */
  void free();

  /**
   * Create a function on a boundary group.
   *
   * @param group ID of boundary group to create function on.
   * @param function_space Family of function to create, e.g. "DG".
   * @param polynomial_order Order of function to create, e.g. 0.
   * @returns Function object on boundary.
   */
  CartesianHMeshFunctionSharedPtr
  create_function(const int group, const std::string function_space,
                  const int polynomial_order);

  /**
   * Project particle data onto a function defined on the surface. Uses the
   * standardarised boundary interface on the sub group.
   *
   * @param particle_sub_group ParticleSubGroup to project onto function.
   * @param sym Sym<REAL> Particle property to use as source weights.
   * @param component Component of particle property to use as source weights.
   * @param is_ephemeral Indicate if the particle weights are in an EphemeralDat
   * or ParticleDat.
   * @param func Function to project onto.
   */
  void function_project(ParticleSubGroupSharedPtr particle_sub_group,
                        Sym<REAL> sym, const int component,
                        const bool is_ephemeral,
                        CartesianHMeshFunctionSharedPtr func);
};

extern template std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration_inner(
    std::shared_ptr<ParticleGroup> particles);
extern template std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration_inner(
    std::shared_ptr<ParticleSubGroup> particles);

extern template void CartesianTrajectoryIntersection::pre_integration_inner(
    std::shared_ptr<ParticleGroup> particles);
extern template void CartesianTrajectoryIntersection::pre_integration_inner(
    std::shared_ptr<ParticleSubGroup> particles);

} // namespace NESO::Particles

#endif
