#ifndef _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_2D_HPP_
#define _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_2D_HPP_

#include "../../../boundary_interaction_specification.hpp"
#include "../../../containers/blocked_binary_tree.hpp"
#include "../../../device_functions.hpp"
#include "boundary_interaction_common.hpp"
#include <cmath>
#include <numeric>

namespace NESO::Particles::PetscInterface {

/**
 * Type to point to the boundary elements that intersect a MeshHierarchy cell.
 */
struct BoundaryInteractionCellData2D {
  int num_edges;
  REAL *d_real;
  int *d_int;
};

/**
 * Type to point to the normal vector for a boundary element.
 */
struct BoundaryInteractionNormalData2D {
  REAL *d_normal;
};

/**
 * Device type to help finding the normal vector for a particle-boundary
 * interaction.
 */
struct BoundaryNormalMapper {
  // Root of the tree containing normal data.
  BlockedBinaryNode<INT, BoundaryInteractionNormalData2D, 8> *root;

  /**
   * Get a pointer to the normal data for an edge element.
   *
   * @param[in] global_id Global index of boundary element to retrive normal
   * for.
   * @param[in, out] normal Pointer to populate with address of normal vector.
   * @returns True if the boundary element is found otherwise false.
   */
  inline bool get(const INT global_id, REAL **normal) const {
    BoundaryInteractionNormalData2D *node = nullptr;
    bool *exists;
    const bool e = root->get_location(global_id, &exists, &node);
    *normal = node->d_normal;
    return e && (*exists);
  }
};

/**
 * Implementation of identifying the intersection of particle trajectories of
 * 2D meshes (with 1D) boundaries. For an instance of this class named b2d
 * users should call @ref pre_integration prior to modifying particle positions
 * and @ref post_integration after modifying particles positions. The
 * constructor and free calls must be collective on the communicator.
 *
 * For creating particle loops which interact with the boundary the normal
 * vector for each boundary element can be identified via the helper struct
 * returned from @ref get_device_normal_mapper.
 */
class BoundaryInteraction2D : public BoundaryInteractionCommon {
protected:
  // An edge has two vertices and each vertex has a coordinate in 2D. Then the
  // normal vector.
  static constexpr int ncomp_real = 2 * 2 + 2;

  // label id, global edge point index
  static constexpr int ncomp_int = 2;

  static constexpr REAL padding = 1.0e-8;

  std::map<INT, std::set<int>> map_mh_index_to_index;

  int num_facets_global;
  MPI_Win facets_win_real;
  REAL *facets_base_real = nullptr;
  REAL *facets_real = nullptr;
  MPI_Win facets_win_int;
  int *facets_base_int = nullptr;
  int *facets_int = nullptr;

  ExternalCommon::BoundingBoxSharedPtr get_bounding_box(const int index);

  std::stack<std::shared_ptr<BufferDevice<REAL>>> stack_d_real;
  std::stack<std::shared_ptr<BufferDevice<int>>> stack_d_int;
  std::set<int> pushed_edge_data;

  std::shared_ptr<BlockedBinaryTree<INT, BoundaryInteractionCellData2D, 8>>
      d_map_edge_discovery;
  std::shared_ptr<BlockedBinaryTree<INT, BoundaryInteractionNormalData2D, 8>>
      d_map_edge_normals;

  void collect_cells();

  struct TrajectoryIntersect2D {
    REAL max_distance;
    REAL epsilon;
    BlockedBinaryNode<INT, BoundaryInteractionCellData2D, 8> *root;
    BlockedBinaryNode<INT, BoundaryInteractionNormalData2D, 8> *root_normals;
    REAL tol;

    inline bool boundary_elements_exist() const {
      return this->root != nullptr;
    }

    inline void reset(REAL &current_distance) const {
      current_distance = max_distance;
    }

    inline void find(const INT linear_cell, const REAL *a, const REAL *b,
                     REAL &current_distance, REAL *P, REAL *NORMAL,
                     INT *C) const {

      bool *exists;
      BoundaryInteractionCellData2D *data = nullptr;
      if (root->get_location(linear_cell, &exists, &data)) {
        if (*exists) {
          const REAL xa = a[0];
          const REAL ya = a[1];
          const REAL xb = b[0];
          const REAL yb = b[1];
          REAL xi, yi, l0;
          bool new_intersection_found = false;

          for (int edgex = 0; edgex < (data->num_edges); edgex++) {
            const REAL x0 = data->d_real[edgex * 4 + 0];
            const REAL y0 = data->d_real[edgex * 4 + 1];
            const REAL x1 = data->d_real[edgex * 4 + 2];
            const REAL y1 = data->d_real[edgex * 4 + 3];
            const INT group_id = data->d_int[edgex * 2 + 0];
            const INT edge_id = data->d_int[edgex * 2 + 1];

            const bool intersects = line_segment_intersection_2d(
                x0, y0, x1, y1, xa, ya, xb, yb, xi, yi, l0, tol);

            const REAL xd = a[0] - xi;
            const REAL yd = a[1] - yi;
            const REAL d2 = xd * xd + yd * yd;

            if (intersects && (d2 < current_distance)) {
              P[0] = xi;
              P[1] = yi;
              C[0] = 1;
              C[1] = group_id;
              C[2] = edge_id;
              current_distance = d2;
              new_intersection_found = true;
            }
          }
          // Populate the normal information for the edge that was found.
          if (new_intersection_found) {
            BoundaryInteractionNormalData2D *normal_data = nullptr;
            if (root_normals->get_location(C[2], &exists, &normal_data)) {
              NORMAL[0] = normal_data->d_normal[0];
              NORMAL[1] = normal_data->d_normal[1];
            }
          }
        }
      }
    }
  };

  template <typename T>
  [[nodiscard]] inline std::map<PetscInt, ParticleSubGroupSharedPtr>
  post_integration_inner(std::shared_ptr<T> particles) {

    this->find_cells(particles);
    this->collect_cells();

    TrajectoryIntersect2D intersect_object;
    intersect_object.max_distance = std::numeric_limits<REAL>::max();
    intersect_object.epsilon = std::numeric_limits<REAL>::min();
    intersect_object.root = this->d_map_edge_discovery->root;
    intersect_object.root_normals = this->d_map_edge_normals->root;
    intersect_object.tol = this->tol;

    auto particle_group = get_particle_group(particles);
    const INT npart_local = particle_group->get_npart_local();

    auto d_real = get_resource<BufferDevice<REAL>,
                               ResourceStackInterfaceBufferDevice<REAL>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
        sycl_target);
    auto d_int = get_resource<BufferDevice<INT>,
                              ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);

    d_real->realloc_no_copy(npart_local * 4);
    d_int->realloc_no_copy(npart_local * 3);
    auto k_real = d_real->ptr;
    auto k_int = d_int->ptr;

    if (npart_local > 0) {
      this->sycl_target->queue.fill(k_int, 0, 3 * npart_local).wait_and_throw();
    }

    this->find_intersections_inner(particles, intersect_object, k_real, k_int);

    std::map<PetscInt, ParticleSubGroupSharedPtr> m;

    auto lambda_make_sub_group = [&](auto iteration_set) {
      for (auto &group_labels : this->boundary_groups) {
        const PetscInt k_group = group_labels.first;
        m[k_group] = static_particle_sub_group(
            iteration_set,
            [=](auto INDEX) -> bool {
              nprint("making sub group:", k_int[INDEX.get_local_linear_index() * 3 + 0], k_int[INDEX.get_local_linear_index() * 3 + 1]);
              return (k_int[INDEX.get_local_linear_index() * 3 + 0]) &&
                     (k_int[INDEX.get_local_linear_index() * 3 + 1] == k_group);
            },
            Access::read(ParticleLoopIndex{}));

        // We can now create the EphemeralDats that describe the standard
        // boundary.
        add_boundary_interaction_ephemeral_dats(m[k_group], 2);

        // Assemble the standard boundary data
        particle_loop(
            "BoundaryInteraction2D::find_intersections_1", m[k_group],
            [=](auto INDEX, auto INTERSECTION_POINT, auto INTERSECTION_NORMAL,
                auto INTERSECTION_METADATA) {
              const INT index = INDEX.get_local_linear_index();
              INTERSECTION_POINT.at_ephemeral(0) = k_real[index * 4 + 0];
              INTERSECTION_POINT.at_ephemeral(1) = k_real[index * 4 + 1];
              INTERSECTION_NORMAL.at_ephemeral(0) = k_real[index * 4 + 2];
              INTERSECTION_NORMAL.at_ephemeral(1) = k_real[index * 4 + 3];
              nprint("Writing EphemeralDatInt:", k_group, k_int[index * 3 + 1], k_int[index * 3 + 2]);
              INTERSECTION_METADATA.at_ephemeral(0) = k_group;
              INTERSECTION_METADATA.at_ephemeral(1) = k_int[index * 3 + 2];
            },
            Access::read(ParticleLoopIndex{}),
            Access::write(BoundaryInteractionSpecification::intersection_point),
            Access::write(
                BoundaryInteractionSpecification::intersection_normal),
            Access::write(
                BoundaryInteractionSpecification::intersection_metadata))
            ->execute();
      }
    };

    // If there is more than one boundary group it is probably more efficient
    // to find all particles hitting the boundary then sub-divide that group
    // into the output groups.
    if (this->boundary_groups.size() == 1) {
      lambda_make_sub_group(particles);
    } else {
      lambda_make_sub_group(static_particle_sub_group(
          particles,
          [=](auto INDEX) -> bool {
            return k_int[INDEX.get_local_linear_index() * 3 + 0] != 0;
          },
          Access::read(ParticleLoopIndex{})));
    }

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<REAL>{}, d_real);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_int);
    return m;
  }

public:
  /// Tolerance for line-line intersections.
  REAL tol;

  /**
   * Get a device callable mapper to map from global boundary indices to normal
   * vectors. This method should be called after @ref post_integration as the
   * map is populated with potentially relvant normal data on the fly.
   *
   * @returns Device copyable and callable mapper.
   */
  BoundaryNormalMapper get_device_normal_mapper();

  /**
   * Free the instance. Must be called. Collective on the communicator.
   */
  void free();

  /**
   * Call after updating to find particles whose trajectories intersect the
   * DMPlex boundary.
   *
   * @param particles Collection of particles, either a ParticleGroup or
   * ParticleSubGroup, to identify trajectory-boundary intersections of.
   * @returns Map from boundary groups ids, which were passed in the
   * constructor, to a ParticleSubGroup of particles which crossed the boundary
   * elements which form the boundary group.
   */
  [[nodiscard]] std::map<PetscInt, ParticleSubGroupSharedPtr>
  post_integration(std::shared_ptr<ParticleGroup> particles);

  /**
   * Call after updating to find particles whose trajectories intersect the
   * DMPlex boundary.
   *
   * @param particles Collection of particles, either a ParticleGroup or
   * ParticleSubGroup, to identify trajectory-boundary intersections of.
   * @returns Map from boundary groups ids, which were passed in the
   * constructor, to a ParticleSubGroup of particles which crossed the boundary
   * elements which form the boundary group.
   */
  [[nodiscard]] std::map<PetscInt, ParticleSubGroupSharedPtr>
  post_integration(std::shared_ptr<ParticleSubGroup> particles);

  /**
   * Create an instance of the class for a particular mesh. This constructor
   * must be called collectively on the communicator of the mesh.
   *
   * @param sycl_target Compute device to use to identify intersections of
   * trajectories and the boundary.
   * @param mesh 2D DMPlex mesh interface to use.
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
  BoundaryInteraction2D(
      SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
      std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
      const REAL tol = 0.0,
      std::optional<Sym<REAL>> previous_position_sym = std::nullopt);
};

extern template std::map<PetscInt, ParticleSubGroupSharedPtr>
BoundaryInteraction2D::post_integration_inner<ParticleGroup>(
    std::shared_ptr<ParticleGroup> particles);

extern template std::map<PetscInt, ParticleSubGroupSharedPtr>
BoundaryInteraction2D::post_integration_inner<ParticleSubGroup>(
    std::shared_ptr<ParticleSubGroup> particles);

} // namespace NESO::Particles::PetscInterface

#endif
