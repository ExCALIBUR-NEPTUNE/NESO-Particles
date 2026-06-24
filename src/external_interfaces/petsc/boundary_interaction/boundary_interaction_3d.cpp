#ifdef NESO_PARTICLES_PETSC

#include <neso_particles/common_impl.hpp>
#include <neso_particles/external_interfaces/petsc/boundary_interaction/boundary_interaction_3d.hpp>

namespace NESO::Particles::PetscInterface {

ExternalCommon::BoundingBoxSharedPtr BoundaryInteraction3D::get_bounding_box(
    const BoundaryInteraction3DTriangle &triangle) {
  auto bb = std::make_shared<ExternalCommon::BoundingBox>();
  std::vector<REAL> bbv(6);

  for (int vx = 0; vx < 3; vx++) {
    auto x = triangle.vertices[vx][0];
    auto y = triangle.vertices[vx][1];
    auto z = triangle.vertices[vx][2];
    bbv.at(0) = x - this->padding;
    bbv.at(1) = y - this->padding;
    bbv.at(2) = z - this->padding;
    bbv.at(3) = x + this->padding;
    bbv.at(4) = y + this->padding;
    bbv.at(5) = z + this->padding;
    auto bbt = std::make_shared<ExternalCommon::BoundingBox>(bbv);
    bb->expand(bbt);
  }

  return bb;
}

void BoundaryInteraction3D::collect_cells() {

  {
    std::vector<INT> gather_cells;
    gather_cells.reserve(this->required_mh_cells.size());
    for (auto &cellx : this->required_mh_cells) {
      gather_cells.push_back(cellx);
    }
    this->mesh_hierarchy_data_triangles->gather(gather_cells);
    gather_cells.clear();
  }

  std::vector<
      MeshHierarchyData::GenericSerialContainer<BoundaryInteraction3DTriangle>>
      triangles;

  std::vector<sycl::marray<REAL, 3>> h_real;
  std::vector<int> h_int;
  for (auto cell : this->required_mh_cells) {

    this->mesh_hierarchy_data_triangles->get(cell, triangles);
    // Does the mh cell actually have any edges intersecting it?

    const std::size_t num_triangles = triangles.size();
    if (num_triangles > 0) {
      // get the real and int data for the mh cell

      h_real.resize(num_triangles * 3);
      h_int.resize(num_triangles * 2);

      for (std::size_t tx = 0; tx < num_triangles; tx++) {

        const BoundaryInteraction3DTriangle &triangle = triangles.at(tx).obj;

        for (int vx = 0; vx < 3; vx++) {
          sycl::marray<REAL, 3> tmp_array{triangle.vertices[vx][0],
                                          triangle.vertices[vx][1],
                                          triangle.vertices[vx][2]};
          h_real.at(tx * 3 + vx) = tmp_array;
        }

        const PetscInt label_id = triangle.label_id;
        const auto group_id = this->map_label_to_groups.at(label_id);
        const auto face_id = triangle.face_id;
        h_int.at(tx * 2 + 0) = group_id;
        h_int.at(tx * 2 + 1) = face_id;

        if (this->pushed_facet_data.count(face_id) == 0) {
          std::vector<REAL> h_norm(3);
          for (int dx = 0; dx < 3; dx++) {
            h_norm.at(dx) = triangle.normal[dx];
          }
          auto t_norm =
              std::make_shared<BufferDevice<REAL>>(this->sycl_target, h_norm);
          this->stack_d_real.push(t_norm);
          BoundaryInteractionNormalData3D dnorm;
          dnorm.d_normal = t_norm->ptr;
          this->d_map_facet_normals->add(face_id, dnorm);
          this->pushed_facet_data.insert(face_id);
        }
      }

      // push cell data onto device
      auto t_real = std::make_shared<BufferDevice<sycl::marray<REAL, 3>>>(
          this->sycl_target, h_real);
      auto t_int =
          std::make_shared<BufferDevice<int>>(this->sycl_target, h_int);

      this->stack_void.push(t_real);
      this->stack_d_int.push(t_int);

      BoundaryInteractionCellData3D d;
      d.num_facets = num_triangles;
      d.d_real = t_real->ptr;
      d.d_int = t_int->ptr;
      this->d_map_facet_discovery->add(cell, d);
    }
    this->collected_mh_cells.insert(cell);
  }
}

BoundaryNormalMapper3D BoundaryInteraction3D::get_device_normal_mapper() {
  BoundaryNormalMapper3D mapper;
  mapper.root = this->d_map_facet_normals->root;
  return mapper;
}

void BoundaryInteraction3D::free() {
  this->mesh_hierarchy_data_triangles->free();
}

std::map<PetscInt, ParticleSubGroupSharedPtr>
BoundaryInteraction3D::post_integration(
    std::shared_ptr<ParticleGroup> particles) {
  return this->post_integration_inner(particles);
}

std::map<PetscInt, ParticleSubGroupSharedPtr>
BoundaryInteraction3D::post_integration(
    std::shared_ptr<ParticleSubGroup> particles) {
  return this->post_integration_inner(particles);
}

BoundaryInteraction3D::BoundaryInteraction3D(
    SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
    std::map<PetscInt, std::vector<PetscInt>> &boundary_groups, const REAL tol,
    std::optional<Sym<REAL>> previous_position_sym)
    : BoundaryInteractionCommon(sycl_target, mesh, boundary_groups,
                                previous_position_sym)

{
  this->tol = tol;

  // Get the boundary labels this instance should detect interactions with.
  auto labels = this->get_labels();

  // map from label to petsc point indices in the dm for the facets
  auto face_sets = this->mesh->dmh->get_face_sets();

  // Keep and flatten the points/labels of interest
  std::vector<PetscInt> facet_labels;
  std::vector<PetscInt> facet_indices;

  int num_triangles_local = 0;
  for (auto &item : face_sets) {
    if (labels.count(item.first)) {
      facet_labels.reserve(facet_labels.size() + item.second.size());
      facet_indices.reserve(facet_indices.size() + item.second.size());
      for (auto &fx : item.second) {
        // push back the label
        facet_labels.push_back(item.first);
        // push back the petsc point index
        facet_indices.push_back(fx);

        // If the facet is a quad then we will split that quad into two
        // triangles.
        const auto cell_type = this->mesh->dmh->get_point_type(fx);
        num_triangles_local += cell_type == DM_POLYTOPE_TRIANGLE ? 1 : 2;
      }
    }
  }
  face_sets.clear();

  int num_facets_local = facet_labels.size();

  std::map<INT, std::vector<MeshHierarchyData::GenericSerialContainer<
                    BoundaryInteraction3DTriangle>>>
      staged_mesh_hierarchy_data;

  // collect the local edges to send
  std::vector<std::vector<REAL>> coords;
  std::deque<std::pair<INT, double>> cells;
  std::vector<REAL> normal_vector;
  auto mesh_hierarchy = this->mesh->get_mesh_hierarchy();
  for (int ix = 0; ix < num_facets_local; ix++) {
    const PetscInt index = facet_indices.at(ix);
    // Collect the vertex coords
    this->mesh->dmh->get_generic_vertices(index, coords);
    NESOASSERT(coords.size() == 3 || coords.size() == 4,
               "Expected a facet to only have three or four vertices.");
    NESOASSERT(coords.at(0).size() == 3 && coords.at(1).size() == 3 &&
                   coords.at(2).size() == 3,
               "Expected face vertex to be embedded in 3D.");

    this->mesh->dmh->get_linear_normal_vector(index, normal_vector);

    BoundaryInteraction3DTriangle triangle_data0;
    BoundaryInteraction3DTriangle triangle_data1;
    const PetscInt facet_global_id =
        this->mesh->dmh->get_point_global_index(index);

    ExternalCommon::BoundingBoxSharedPtr bounding_box = nullptr;

    bool is_triangle = false;

    auto lambda_set_common = [&](auto &triangle) {
      triangle.label_id = facet_labels.at(ix);
      triangle.face_id = facet_global_id;
      for (int dx = 0; dx < 3; dx++) {
        triangle.normal[dx] = normal_vector[dx];
      }
    };

    if (coords.size() == 3) {
      is_triangle = true;
      for (int cx = 0; cx < 3; cx++) {
        for (int dx = 0; dx < 3; dx++) {
          triangle_data0.vertices[cx][dx] = coords.at(cx).at(dx);
        }
      }
      lambda_set_common(triangle_data0);
      bounding_box = this->get_bounding_box(triangle_data0);
    } else {
      is_triangle = false;
      std::array<std::array<PetscInt, 3>, 2> triangle_indices;
      split_quadrilateral_into_two_triangles(this->mesh->dmh->dm, index,
                                             triangle_indices);

      for (int cx = 0; cx < 3; cx++) {
        const PetscInt vx = triangle_indices.at(0).at(cx);
        this->mesh->dmh->get_generic_vertices(vx, coords);
        NESOASSERT(coords.size() == 1, "Expect coords to be size 1.");

        for (int dx = 0; dx < 3; dx++) {
          triangle_data0.vertices[cx][dx] = coords.at(0).at(dx);
        }
      }
      lambda_set_common(triangle_data0);

      for (int cx = 0; cx < 3; cx++) {
        const PetscInt vx = triangle_indices.at(1).at(cx);
        this->mesh->dmh->get_generic_vertices(vx, coords);
        NESOASSERT(coords.size() == 1, "Expect coords to be size 1.");

        for (int dx = 0; dx < 3; dx++) {
          triangle_data1.vertices[cx][dx] = coords.at(0).at(dx);
        }
      }
      lambda_set_common(triangle_data1);

      bounding_box = this->get_bounding_box(triangle_data0);
      bounding_box->expand(this->get_bounding_box(triangle_data1));
    }

    cells.clear();
    ExternalCommon::bounding_box_map(bounding_box, mesh_hierarchy, cells);
    for (auto &cx_w : cells) {
      if (is_triangle) {
        staged_mesh_hierarchy_data[cx_w.first].emplace_back(triangle_data0);
      } else {
        staged_mesh_hierarchy_data[cx_w.first].emplace_back(triangle_data0);
        staged_mesh_hierarchy_data[cx_w.first].emplace_back(triangle_data1);
      }
    }
  }

  this->mesh_hierarchy_data_triangles =
      std::make_shared<MeshHierarchyData::MeshHierarchyContainer<
          MeshHierarchyData::GenericSerialContainer<
              BoundaryInteraction3DTriangle>>>(mesh_hierarchy,
                                               staged_mesh_hierarchy_data);
  staged_mesh_hierarchy_data.clear();

  this->d_map_facet_discovery = std::make_shared<
      BlockedBinaryTree<INT, BoundaryInteractionCellData3D,
                        NESO_PARTICLES_BLOCKED_BINARY_TREE_WIDTH>>(
      this->sycl_target);

  this->d_map_facet_normals = std::make_shared<
      BlockedBinaryTree<INT, BoundaryInteractionNormalData3D,
                        NESO_PARTICLES_BLOCKED_BINARY_TREE_WIDTH>>(
      this->sycl_target);
}

template std::map<PetscInt, ParticleSubGroupSharedPtr>
BoundaryInteraction3D::post_integration_inner<ParticleGroup>(
    std::shared_ptr<ParticleGroup> particles);

template std::map<PetscInt, ParticleSubGroupSharedPtr>
BoundaryInteraction3D::post_integration_inner<ParticleSubGroup>(
    std::shared_ptr<ParticleSubGroup> particles);

} // namespace NESO::Particles::PetscInterface

#endif
