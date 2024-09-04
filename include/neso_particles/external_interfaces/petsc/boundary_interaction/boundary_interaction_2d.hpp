#ifndef _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_2D_HPP_
#define _NESO_PARTICLES_PETSC_BOUNDARY_INTERACTION_BOUNDARY_INTERACTION_2D_HPP_

#include "../../../containers/blocked_binary_tree.hpp"
#include "../../../device_functions.hpp"
#include "../../common/local_claim.hpp"
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
    BoundaryInteractionNormalData2D *node;
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

  inline ExternalCommon::BoundingBoxSharedPtr
  get_bounding_box(const int index) {
    NESOASSERT(this->facets_real != nullptr, "Expected a non-nullptr.");
    auto bb = std::make_shared<ExternalCommon::BoundingBox>();
    std::vector<REAL> bbv(6);
    bbv.at(2) = 0.0;
    bbv.at(5) = 0.0;

    for (int vx = 0; vx < 2; vx++) {
      auto x = this->facets_real[index * this->ncomp_real + vx * 2 + 0];
      auto y = this->facets_real[index * this->ncomp_real + vx * 2 + 1];
      bbv.at(0) = x - this->padding;
      bbv.at(1) = y - this->padding;
      bbv.at(3) = x + this->padding;
      bbv.at(4) = y + this->padding;
      auto bbt = std::make_shared<ExternalCommon::BoundingBox>(bbv);
      bb->expand(bbt);
    }

    return bb;
  }

  std::stack<std::shared_ptr<BufferDevice<REAL>>> stack_d_real;
  std::stack<std::shared_ptr<BufferDevice<int>>> stack_d_int;
  std::set<int> pushed_edge_data;

  std::shared_ptr<BlockedBinaryTree<INT, BoundaryInteractionCellData2D, 8>>
      d_map_edge_discovery;
  std::shared_ptr<BlockedBinaryTree<INT, BoundaryInteractionNormalData2D, 8>>
      d_map_edge_normals;

  inline void collect_cells() {
    for (auto cell : this->required_mh_cells) {
      // Does the mh cell actually have any edges intersecting it?
      if (this->map_mh_index_to_index.count(cell)) {
        const int num_edges = this->map_mh_index_to_index.at(cell).size();

        // get the real and int data for the mh cell
        std::vector<REAL> h_real(num_edges * 4);
        std::vector<int> h_int(num_edges * ncomp_int);
        int index = 0;
        for (auto ix : this->map_mh_index_to_index.at(cell)) {
          for (int cx = 0; cx < 4; cx++) {
            h_real.at(index * 4 + cx) = this->facets_real[ix * ncomp_real + cx];
          }

          const auto label = this->facets_int[ix * ncomp_int + 0];
          const auto edge_id = this->facets_int[ix * ncomp_int + 1];
          const auto group_id = this->map_label_to_groups.at(label);
          h_int.at(index * ncomp_int + 0) = group_id;
          h_int.at(index * ncomp_int + 1) = edge_id;
          index++;

          // Is this edge in the map of edge data for interactions?
          if (this->pushed_edge_data.count(edge_id) == 0) {
            std::vector<REAL> h_norm(2);
            h_norm.at(0) = this->facets_real[ix * ncomp_real + 4];
            h_norm.at(1) = this->facets_real[ix * ncomp_real + 5];
            auto t_norm =
                std::make_shared<BufferDevice<REAL>>(this->sycl_target, h_norm);
            this->stack_d_real.push(t_norm);
            BoundaryInteractionNormalData2D dnorm;
            dnorm.d_normal = t_norm->ptr;
            this->d_map_edge_normals->add(edge_id, dnorm);
            this->pushed_edge_data.insert(edge_id);
          }
        }

        // push cell data onto device
        auto t_real =
            std::make_shared<BufferDevice<REAL>>(this->sycl_target, h_real);
        auto t_int =
            std::make_shared<BufferDevice<int>>(this->sycl_target, h_int);

        this->stack_d_real.push(t_real);
        this->stack_d_int.push(t_int);

        BoundaryInteractionCellData2D d;
        d.num_edges = num_edges;
        d.d_real = t_real->ptr;
        d.d_int = t_int->ptr;
        this->d_map_edge_discovery->add(cell, d);
      }
      this->collected_mh_cells.insert(cell);
    }
  }

  struct TrajectoryIntersect2D {
    REAL max_distance;
    REAL epsilon;
    BlockedBinaryNode<INT, BoundaryInteractionCellData2D, 8> *root;
    REAL tol;

    inline bool boundary_elements_exist() const {
      return this->root != nullptr;
    }

    inline void reset(REAL &current_distance) const {
      current_distance = max_distance;
    }

    template <typename P_TYPE, typename C_TYPE>
    inline void find(const INT linear_cell, const REAL *a, const REAL *b,
                     REAL &current_distance, P_TYPE &P, C_TYPE &C) const {

      bool *exists;
      BoundaryInteractionCellData2D *data;
      if (root->get_location(linear_cell, &exists, &data)) {
        if (*exists) {
          const REAL xa = a[0];
          const REAL ya = a[1];
          const REAL xb = b[0];
          const REAL yb = b[1];
          REAL xi, yi, l0;

          for (int edgex = 0; edgex < (data->num_edges); edgex++) {
            const REAL x0 = data->d_real[edgex * 4 + 0];
            const REAL y0 = data->d_real[edgex * 4 + 1];
            const REAL x1 = data->d_real[edgex * 4 + 2];
            const REAL y1 = data->d_real[edgex * 4 + 3];
            const INT group_id = data->d_int[edgex * 2 + 0];
            const INT edge_id = data->d_int[edgex * 2 + 1];

            const bool intersects = line_segment_intersection_2d(
                xa, ya, xb, yb, x0, y0, x1, y1, xi, yi, l0, tol);

            if (intersects && (l0 < current_distance)) {
              P.at(0) = xi;
              P.at(1) = yi;
              C.at(0) = 1;
              C.at(1) = group_id;
              C.at(2) = edge_id;
              current_distance = l0;
            }
          }
        }
      }
    }
  };

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
  inline BoundaryNormalMapper get_device_normal_mapper() {
    BoundaryNormalMapper mapper;
    mapper.root = this->d_map_edge_normals->root;
    return mapper;
  }

  /**
   * Free the instance. Must be called. Collective on the communicator.
   */
  inline void free() {
    if (this->facets_real != nullptr) {
      MPICHK(MPI_Win_free(&this->facets_win_real));
      this->facets_base_real = nullptr;
      this->facets_real = nullptr;
    }
    if (this->facets_int != nullptr) {
      MPICHK(MPI_Win_free(&this->facets_win_int));
      this->facets_base_int = nullptr;
      this->facets_int = nullptr;
    }
  }

  /**
   * Called after updating to find particles whose trajectories intersect the
   * DMPlex boundary.
   *
   * @param particles Collection of particles, either a ParticleGroup or
   * ParticleSubGroup, to identify trajectory-boundary intersections of.
   * @returns Map from boundary groups ids, which were passed in the
   * constructor, to a ParticleSubGroup of particles which crossed the boundary
   * elements which form the boundary group.
   */
  template <typename T>
  [[nodiscard]] inline std::map<PetscInt, ParticleSubGroupSharedPtr>
  post_integration(std::shared_ptr<T> particles) {

    this->find_cells(particles);
    this->collect_cells();

    auto particle_group = this->get_particle_group(particles);
    const auto k_ndim = particle_group->position_dat->ncomp;
    particle_loop(
        "BoundaryInteraction2D::find_intersections_0", particles,
        [=](auto C) { C.at(0) = -1; }, Access::write(this->boundary_label_sym))
        ->execute();

    TrajectoryIntersect2D intersect_object;
    intersect_object.max_distance = std::numeric_limits<REAL>::max();
    intersect_object.epsilon = std::numeric_limits<REAL>::min();
    intersect_object.root = this->d_map_edge_discovery->root;
    intersect_object.tol = this->tol;

    this->find_intersections_inner(particles, intersect_object);

    std::map<PetscInt, ParticleSubGroupSharedPtr> m;

    auto lambda_make_sub_group = [&](auto iteration_set) {
      for (auto &group_labels : this->boundary_groups) {
        const PetscInt k_group = group_labels.first;
        m[k_group] = particle_sub_group(
            iteration_set,
            [=](auto C) -> bool {
              return (C.at(0) > -1) && (C.at(1) == k_group);
            },
            Access::read(this->boundary_label_sym));
      }
    };

    // If there is more than one boundary group it is probably more efficient
    // to find all particles hitting the boundary then sub-divide that group
    // into the output groups.
    if (this->boundary_groups.size() == 1) {
      lambda_make_sub_group(particles);
    } else {
      lambda_make_sub_group(particle_sub_group(
          particles, [=](auto C) -> bool { return C.at(0) > -1; },
          Access::read(this->boundary_label_sym)));
    }

    return m;
  }

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
   * @param boundary_position_sym The Sym for the intersection point for the
   * particle trajectory and the boundary. This property is populated if an
   * intersection is discovered in a call to @ref post_integration.
   * @param boundary_label_sym The Sym which holds the metadata information for
   * the intersection point between the trajectory and the boundary. Component 0
   * holds a 1 if an intersection is found. Component 1 holds the group ID
   * identifed for the intersection. Component 2 holds the global ID of the
   * boundary element for which the intersection was identified between
   * trajectory and the boundary.
   *
   */
  BoundaryInteraction2D(
      SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
      std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
      const REAL tol = 0.0,
      std::optional<Sym<REAL>> previous_position_sym = std::nullopt,
      std::optional<Sym<REAL>> boundary_position_sym = std::nullopt,
      std::optional<Sym<INT>> boundary_label_sym = std::nullopt)
      :

        BoundaryInteractionCommon(sycl_target, mesh, boundary_groups,
                                  previous_position_sym, boundary_position_sym,
                                  boundary_label_sym)

  {
    this->tol = tol;

    // Get the boundary labels this instance should detect interactions with.
    auto labels = this->get_labels();

    // map from label to petsc point indices in the dm for the facets
    auto face_sets = this->mesh->dmh->get_face_sets();

    // Keep and flatten the points/labels of interest
    std::vector<PetscInt> facet_labels;
    std::vector<PetscInt> facet_indices;
    for (auto &item : face_sets) {
      if (labels.count(item.first)) {
        facet_labels.reserve(facet_labels.size() + item.second.size());
        facet_indices.reserve(facet_indices.size() + item.second.size());
        for (auto &fx : item.second) {
          // push back the label
          facet_labels.push_back(item.first);
          // push back the petsc point index
          facet_indices.push_back(fx);
        }
      }
    }
    face_sets.clear();

    int num_facets_local = facet_labels.size();

    // space to store the local contributions
    std::vector<REAL> local_real(num_facets_local * ncomp_real);
    std::vector<int> local_int(num_facets_local * ncomp_int);

    // collect the local edges to send
    std::vector<std::vector<REAL>> coords;
    for (int ix = 0; ix < num_facets_local; ix++) {
      const PetscInt index = facet_indices.at(ix);
      // Collect the vertex coords
      this->mesh->dmh->get_generic_vertices(index, coords);
      NESOASSERT(coords.size() == 2,
                 "Expected an edge to only have two vertices.");
      NESOASSERT(coords.at(0).size() == 2,
                 "Expected edge vertex to be embedded in 2D.");
      NESOASSERT(coords.at(1).size() == 2,
                 "Expected edge vertex to be embedded in 2D.");

      const REAL x0 = coords.at(0).at(0);
      const REAL y0 = coords.at(0).at(1);
      const REAL x1 = coords.at(1).at(0);
      const REAL y1 = coords.at(1).at(1);

      // compute the normal to the facet
      const REAL dx = x1 - x0;
      const REAL dy = y1 - y0;
      const REAL n0t = -dy;
      const REAL n1t = dx;
      const REAL l = 1.0 / std::sqrt(n0t * n0t + n1t * n1t);
      const REAL n0 = n0t * l;
      const REAL n1 = n1t * l;

      local_real.at(ix * ncomp_real + 0) = x0;
      local_real.at(ix * ncomp_real + 1) = y0;
      local_real.at(ix * ncomp_real + 2) = x1;
      local_real.at(ix * ncomp_real + 3) = y1;
      local_real.at(ix * ncomp_real + 4) = n0;
      local_real.at(ix * ncomp_real + 5) = n1;

      // collect the label index and edge global id
      const PetscInt facet_global_id =
          this->mesh->dmh->get_point_global_index(index);
      local_int.at(ix * ncomp_int + 0) = facet_labels.at(ix);
      local_int.at(ix * ncomp_int + 1) = facet_global_id;
    }

    facet_labels.clear();
    facet_indices.clear();

    MPI_Comm comm_intra = this->sycl_target->comm_pair.comm_intra;
    MPI_Comm comm_inter = this->sycl_target->comm_pair.comm_inter;
    int rank_intra = this->sycl_target->comm_pair.rank_intra;

    std::vector<REAL> node_real;
    std::vector<int> node_int;

    gather_v(local_real, comm_intra, 0, node_real);
    gather_v(local_int, comm_intra, 0, node_int);
    local_real.clear();
    local_int.clear();

    std::vector<REAL> global_real;
    std::vector<int> global_int;

    int num_facets_global_tmp = 0;
    if (rank_intra == 0) {
      all_gather_v(node_real, comm_inter, global_real);
      all_gather_v(node_int, comm_inter, global_int);
      num_facets_global_tmp = global_int.size() / ncomp_int;
    }
    node_real.clear();
    node_int.clear();

    // Allocate the shared space to store the edges
    MPICHK(MPI_Win_allocate_shared(
        num_facets_global_tmp * ncomp_real * sizeof(REAL), sizeof(REAL),
        MPI_INFO_NULL, comm_intra, (void *)&this->facets_base_real,
        &this->facets_win_real));
    MPICHK(MPI_Win_allocate_shared(
        num_facets_global_tmp * ncomp_int * sizeof(int), sizeof(int),
        MPI_INFO_NULL, comm_intra, (void *)&this->facets_base_int,
        &this->facets_win_int));
    // Get the pointers to the shared space on each rank
    MPI_Aint win_size_tmp;
    int disp_unit_tmp;
    MPICHK(MPI_Win_shared_query(this->facets_win_real, 0, &win_size_tmp,
                                &disp_unit_tmp, (void *)&this->facets_real));
    MPICHK(MPI_Win_shared_query(this->facets_win_int, 0, &win_size_tmp,
                                &disp_unit_tmp, (void *)&this->facets_int));

    // On node rank zero copy the data into the shared region.
    if (rank_intra == 0) {
      std::memcpy(this->facets_real, global_real.data(),
                  num_facets_global_tmp * ncomp_real * sizeof(REAL));
      std::memcpy(this->facets_int, global_int.data(),
                  num_facets_global_tmp * ncomp_int * sizeof(int));
    }
    global_real.clear();
    global_int.clear();

    // On each node the rank where rank_intra == 0 now holds all the boundary
    // edges.
    MPICHK(MPI_Bcast(&num_facets_global_tmp, 1, MPI_INT, 0, comm_intra));
    this->num_facets_global = num_facets_global_tmp;
    // Wait for node rank 0 to populate shared memory
    MPICHK(MPI_Barrier(comm_intra));

    // build map from mesh hierarchy cells to indices in the edge data store
    auto mesh_hierarchy = this->mesh->get_mesh_hierarchy();
    std::deque<std::pair<INT, double>> cells;
    for (int ex = 0; ex < this->num_facets_global; ex++) {
      cells.clear();
      auto bb = this->get_bounding_box(ex);
      ExternalCommon::bounding_box_map(bb, mesh_hierarchy, cells);
      for (auto &cx_w : cells) {
        this->map_mh_index_to_index[cx_w.first].insert(ex);
      }
    }

    this->d_map_edge_discovery = std::make_shared<
        BlockedBinaryTree<INT, BoundaryInteractionCellData2D, 8>>(
        this->sycl_target);
    this->d_map_edge_normals = std::make_shared<
        BlockedBinaryTree<INT, BoundaryInteractionNormalData2D, 8>>(
        this->sycl_target);
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
