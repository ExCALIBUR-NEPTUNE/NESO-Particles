#include <neso_particles/cartesian_mesh/cartesian_trajectory_intersection.hpp>
#include <neso_particles/common_impl.hpp>
#include <neso_particles/error_propagate.hpp>

namespace NESO::Particles {

void CartesianTrajectoryIntersection::setup() {
  auto ndim = this->mesh->get_ndim();
  auto mh = this->mesh->get_mesh_hierarchy();

  std::array<INT, 6> element_offsets_tmp = {0, 0, 0, 0, 0, 0};

  if (ndim == 2) {
    element_offsets_tmp[0] = mh->dims[0] * mh->ncells_dim_fine;
    element_offsets_tmp[1] = mh->dims[1] * mh->ncells_dim_fine;
    element_offsets_tmp[2] = mh->dims[0] * mh->ncells_dim_fine;
    element_offsets_tmp[3] = mh->dims[1] * mh->ncells_dim_fine;

    const INT total_boundary_cells = std::accumulate(
        element_offsets_tmp.begin(), element_offsets_tmp.end(), 0);
    NESOASSERT(total_boundary_cells ==
                   mh->ncells_dim_fine *
                       (mh->dims[0] + mh->dims[1] + mh->dims[0] + mh->dims[1]),
               "Incorrect number of boundary cells.");

    this->element_strides0[0] = mh->dims[0] * mh->ncells_dim_fine;
    this->element_strides0[1] = mh->dims[1] * mh->ncells_dim_fine;
    this->element_strides0[2] = mh->dims[0] * mh->ncells_dim_fine;
    this->element_strides0[3] = mh->dims[1] * mh->ncells_dim_fine;
    this->element_strides1[0] = 1;
    this->element_strides1[1] = 1;
    this->element_strides1[2] = 1;
    this->element_strides1[3] = 1;

  } else {
    element_offsets_tmp[0] =
        mh->dims[0] * mh->dims[2] * std::pow(mh->ncells_dim_fine, 2);
    element_offsets_tmp[1] =
        mh->dims[1] * mh->dims[2] * std::pow(mh->ncells_dim_fine, 2);
    element_offsets_tmp[2] =
        mh->dims[0] * mh->dims[2] * std::pow(mh->ncells_dim_fine, 2);
    element_offsets_tmp[3] =
        mh->dims[1] * mh->dims[2] * std::pow(mh->ncells_dim_fine, 2);
    element_offsets_tmp[4] =
        mh->dims[0] * mh->dims[1] * std::pow(mh->ncells_dim_fine, 2);
    element_offsets_tmp[5] =
        mh->dims[0] * mh->dims[1] * std::pow(mh->ncells_dim_fine, 2);

    this->element_strides0[0] = mh->dims[0] * mh->ncells_dim_fine;
    this->element_strides0[1] = mh->dims[1] * mh->ncells_dim_fine;
    this->element_strides0[2] = mh->dims[0] * mh->ncells_dim_fine;
    this->element_strides0[3] = mh->dims[1] * mh->ncells_dim_fine;
    this->element_strides0[4] = mh->dims[0] * mh->ncells_dim_fine;
    this->element_strides0[5] = mh->dims[0] * mh->ncells_dim_fine;

    this->element_strides1[0] = mh->dims[2] * mh->ncells_dim_fine;
    this->element_strides1[1] = mh->dims[2] * mh->ncells_dim_fine;
    this->element_strides1[2] = mh->dims[2] * mh->ncells_dim_fine;
    this->element_strides1[3] = mh->dims[2] * mh->ncells_dim_fine;
    this->element_strides1[4] = mh->dims[1] * mh->ncells_dim_fine;
    this->element_strides1[5] = mh->dims[1] * mh->ncells_dim_fine;

    const INT total_boundary_cells = std::accumulate(
        element_offsets_tmp.begin(), element_offsets_tmp.end(), 0);
    NESOASSERT(total_boundary_cells ==
                   mh->ncells_dim_fine * mh->ncells_dim_fine *
                       (mh->dims[0] * mh->dims[2] + mh->dims[1] * mh->dims[2] +
                        mh->dims[0] * mh->dims[2] + mh->dims[1] * mh->dims[2] +
                        mh->dims[0] * mh->dims[1] + mh->dims[0] * mh->dims[1]),
               "Incorrect number of boundary cells.");
  }
  std::exclusive_scan(element_offsets_tmp.begin(), element_offsets_tmp.end(),
                      this->element_offsets.begin(), 0);
  this->inverse_cell_width_fine = mh->inverse_cell_width_fine;
}

CartesianTrajectoryIntersection::~CartesianTrajectoryIntersection() {
  this->free();
}

CartesianTrajectoryIntersection::CartesianTrajectoryIntersection(
    SYCLTargetSharedPtr sycl_target, CartesianHMeshSharedPtr mesh,
    std::map<int, std::vector<int>> boundary_groups, REAL tolerance)
    : sycl_target(sycl_target), mesh(mesh), boundary_groups(boundary_groups),
      tolerance(tolerance) {
  const int k_ndim = this->mesh->get_ndim();
  NESOASSERT(k_ndim == 2 || k_ndim == 3,
             "This method is only implemented in 2D and 3D.");
  this->setup();

  std::vector<INT> tmp_face_cells;
  for (const auto &gx : boundary_groups) {
    tmp_face_cells.clear();
    for (auto &fx : gx.second) {
      auto face_cells = mesh->get_all_face_cells_on_face(fx);
      tmp_face_cells.insert(tmp_face_cells.end(), face_cells.begin(),
                            face_cells.end());
    }

    this->map_groups_boundary_interface[gx.first] =
        std::make_shared<BoundaryMeshInterface>(mesh->get_comm(), sycl_target,
                                                tmp_face_cells);
    this->map_groups_unseen_value_extractor[gx.first] =
        std::make_shared<UnseenValueExtractor>(this->sycl_target);
  }
}

void CartesianTrajectoryIntersection::prepare_particle_group(
    ParticleGroupSharedPtr particle_group) {
  this->check_dat(particle_group, this->previous_position_sym,
                  this->mesh->get_ndim());
}

void CartesianTrajectoryIntersection::free() {
  for (const auto &gx : this->map_groups_boundary_interface) {
    gx.second->free();
  }
  this->map_groups_boundary_interface.clear();
}

CartesianHMeshFunctionSharedPtr
CartesianTrajectoryIntersection::create_function(
    const int group, const std::string function_space,
    const int polynomial_order) {

  NESOASSERT(this->boundary_groups.count(group),
             "Passed group does not exist in boundary groups.");

  std::vector<INT> cells;
  for (auto &cx : this->boundary_groups.at(group)) {
    auto tmp = this->mesh->get_all_face_cells_on_face(cx);
    cells.insert(cells.end(), tmp.begin(), tmp.end());
  }

  return std::make_shared<CartesianHMeshFunction>(
      this->mesh, this->sycl_target, this->mesh->get_ndim() - 1, cells,
      function_space, polynomial_order, group);
}

void CartesianTrajectoryIntersection::function_project(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CartesianHMeshFunctionSharedPtr func) {

  const int group = func->element_group;
  auto &boundary_mesh_interface = this->map_groups_boundary_interface.at(group);

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const std::size_t tmp_buffer_size =
      num_accessible_geoms * func->cell_dof_count;
  auto d_buffer = get_resource<BufferDevice<REAL>,
                               ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_buffer->realloc_no_copy(tmp_buffer_size);
  REAL *k_buffer = d_buffer->ptr;

  this->sycl_target->queue.fill(k_buffer, (REAL)0.0, tmp_buffer_size)
      .wait_and_throw();

  auto *k_tree_root = d_tree_root;
  NESOASSERT(particle_sub_group->contains_ephemeral_dat(
                 Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
             "Boundary metadata not found on ParticleSubGroup.");
  NESOASSERT(
      (get_particle_group(particle_sub_group)->contains_dat(sym) &&
       (!is_ephemeral)) ||
          (particle_sub_group->contains_ephemeral_dat(sym) && is_ephemeral),
      "Source particle data not found.");

  ErrorPropagate ep(this->sycl_target);
  auto k_ep = ep.device_ptr();

  if (is_ephemeral) {
    particle_loop(
        "CartesianHMeshFunction::function_project", particle_sub_group,
        [=](auto BOUNDARY_METADATA, auto SYM) {
          if (k_tree_root != nullptr) {
            const INT *index;
            const bool found =
                k_tree_root->get(BOUNDARY_METADATA.at_ephemeral(1), &index);
            NESO_KERNEL_ASSERT(found, k_ep);
            if (found) {
              atomic_fetch_add(&k_buffer[*index], SYM.at_ephemeral(component));
            }
          }
        },
        Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
        Access::read(sym))
        ->execute();
  } else {
    particle_loop(
        "CartesianHMeshFunction::function_project", particle_sub_group,
        [=](auto BOUNDARY_METADATA, auto SYM) {
          if (k_tree_root != nullptr) {
            const INT *index;
            const bool found =
                k_tree_root->get(BOUNDARY_METADATA.at_ephemeral(1), &index);
            NESO_KERNEL_ASSERT(found, k_ep);
            if (found) {
              atomic_fetch_add(&k_buffer[*index], SYM.at(component));
            }
          }
        },
        Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
        Access::read(sym))
        ->execute();
  }
  NESOASSERT(!ep.get_flag(), "Failed to find index for hit geometry object.");

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_buffer);
}

template void CartesianTrajectoryIntersection::pre_integration_inner(
    std::shared_ptr<ParticleGroup> particles);
template void CartesianTrajectoryIntersection::pre_integration_inner(
    std::shared_ptr<ParticleSubGroup> particles);

template std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration_inner(
    std::shared_ptr<ParticleGroup> particles);
template std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration_inner(
    std::shared_ptr<ParticleSubGroup> particles);

void CartesianTrajectoryIntersection::pre_integration(
    std::shared_ptr<ParticleGroup> particles) {
  return this->pre_integration_inner(particles);
}

void CartesianTrajectoryIntersection::pre_integration(
    std::shared_ptr<ParticleSubGroup> particles) {
  return this->pre_integration_inner(particles);
}

std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration(
    std::shared_ptr<ParticleGroup> particles) {
  return this->post_integration_inner(particles);
}

std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration(
    std::shared_ptr<ParticleSubGroup> particles) {
  return this->post_integration_inner(particles);
}

} // namespace NESO::Particles
