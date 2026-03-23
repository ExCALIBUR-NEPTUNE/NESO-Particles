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
    NESOASSERT(gx.first != Private::CART_TRAJ_INT_MASK_VALUE,
               "This boundary group label is reseverved.");
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

  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection", "create_function");

  NESOASSERT(this->boundary_groups.count(group),
             "Passed group does not exist in boundary groups.");

  std::vector<INT> cells;
  for (auto &cx : this->boundary_groups.at(group)) {
    auto tmp = this->mesh->get_all_face_cells_on_face(cx);
    cells.insert(cells.end(), tmp.begin(), tmp.end());
  }

  auto func = std::make_shared<CartesianHMeshFunction>(
      this->mesh, this->sycl_target, this->mesh->get_ndim() - 1, cells,
      function_space, polynomial_order, group);

  this->sycl_target->profile_map.end_region(r0);
  return func;
}

void CartesianTrajectoryIntersection::function_project_initialise(
    CartesianHMeshFunctionSharedPtr func) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection", "function_project_initialise");

  const int group = func->boundary_group;
  auto &boundary_mesh_interface = this->map_groups_boundary_interface.at(group);

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const std::size_t tmp_buffer_size =
      num_accessible_geoms * func->cell_dof_count;

  func->d_dofs_stage->realloc_no_copy(tmp_buffer_size);
  REAL *k_buffer = func->d_dofs_stage->ptr;

  if (tmp_buffer_size > 0) {
    this->sycl_target->queue.fill(k_buffer, (REAL)0.0, tmp_buffer_size)
        .wait_and_throw();
  }
  func->fill(0.0);

  this->sycl_target->profile_map.end_region(r0);
}

void CartesianTrajectoryIntersection::function_project_contribute(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CartesianHMeshFunctionSharedPtr func) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection", "function_project_contribute");

  const bool null_sub_group = particle_sub_group == nullptr;
  const int group = func->boundary_group;
  auto &boundary_mesh_interface = this->map_groups_boundary_interface.at(group);

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const INT k_num_accessible_geoms = num_accessible_geoms;

  const auto cell_dof_count = func->cell_dof_count;
  const INT current_stage_size = func->d_dofs_stage->size / cell_dof_count;
  if (current_stage_size < k_num_accessible_geoms) {

    func->d_dofs_stage->realloc(k_num_accessible_geoms * cell_dof_count);
    const auto diff =
        (k_num_accessible_geoms - current_stage_size) * cell_dof_count;
    REAL *k_buffer =
        func->d_dofs_stage->ptr + current_stage_size * cell_dof_count;
    this->sycl_target->queue
        .parallel_for(sycl::range<1>(diff),
                      [=](auto idx) { k_buffer[idx] = 0.0; })
        .wait_and_throw();
  }

  NESOASSERT(
      func->d_dofs_stage->size >=
          static_cast<std::size_t>(num_accessible_geoms * cell_dof_count),
      "Temporary staging buffer is too small. Please raise an issue.");

  REAL *k_buffer = func->d_dofs_stage->ptr;

  if (!null_sub_group) {
    auto *k_tree_root = d_tree_root;
    NESOASSERT(particle_sub_group->contains_ephemeral_dat(
                   Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
               "Boundary metadata not found on ParticleSubGroup.");
    NESOASSERT(
        (get_particle_group(particle_sub_group)->contains_dat(sym) &&
         (!is_ephemeral)) ||
            (particle_sub_group->contains_ephemeral_dat(sym) && is_ephemeral),
        "Source particle data not found.");

    ErrorPropagate ep_found(this->sycl_target);
    ErrorPropagate ep_dof(this->sycl_target);
    auto k_ep_found = ep_found.device_ptr();
    auto k_ep_dof = ep_dof.device_ptr();
    const REAL k_inverse_width = func->mesh->inverse_cell_width_fine;

    auto lambda_dispatch = [&](auto extract_quantity) {
      particle_loop(
          "CartesianHMeshFunction::function_project_contribute",
          particle_sub_group,
          [=](auto BOUNDARY_METADATA, auto SYM) {
            if (k_tree_root != nullptr) {
              const INT *index;
              bool found = false;
              found =
                  k_tree_root->get(BOUNDARY_METADATA.at_ephemeral(1), &index);
#ifndef NDEBUG
              NESO_KERNEL_ASSERT(found, k_ep_found);
              const bool bad_index =
                  ((*index) < 0) || ((*index) >= k_num_accessible_geoms);
              NESO_KERNEL_ASSERT(!bad_index, k_ep_dof);
#endif
              if (found) {
                const REAL value = extract_quantity(SYM, component);
                atomic_fetch_add(&k_buffer[*index], k_inverse_width * value);
              }
            }
          },
          Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
          Access::read(sym))
          ->execute();
    };

    if (is_ephemeral) {
      lambda_dispatch([](auto SYM, const int component) {
        return SYM.at_ephemeral(component);
      });
    } else {
      lambda_dispatch(
          [](auto SYM, const int component) { return SYM.at(component); });
    }
    NESOASSERT(!ep_found.get_flag(),
               "Failed to find index for hit geometry object.");
    NESOASSERT(!ep_dof.get_flag(), "Bad index for hit geometry object.");
  }

  this->sycl_target->profile_map.end_region(r0);
}

void CartesianTrajectoryIntersection::function_project_finalise(
    CartesianHMeshFunctionSharedPtr func) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection", "function_project_finalise");

  const int group = func->boundary_group;
  auto &boundary_mesh_interface = this->map_groups_boundary_interface.at(group);
  boundary_mesh_interface->exchange_from_device(
      func->d_dofs_stage->ptr, func->cell_dof_count, func->d_dofs->ptr);
  func->reset_version();
  NESOASSERT(func->version == 0, "Expected a version reset.");

  this->sycl_target->profile_map.end_region(r0);
}

void CartesianTrajectoryIntersection::function_project(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CartesianHMeshFunctionSharedPtr func) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection", "function_project");

  this->function_project_initialise(func);
  this->function_project_contribute(particle_sub_group, sym, component,
                                    is_ephemeral, func);
  this->function_project_finalise(func);

  this->sycl_target->profile_map.end_region(r0);
}

void CartesianTrajectoryIntersection::function_evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    CartesianHMeshFunctionSharedPtr func) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection", "function_evaluate");

  const bool null_sub_group = particle_sub_group == nullptr;
  const int group = func->boundary_group;
  auto &boundary_mesh_interface = this->map_groups_boundary_interface.at(group);

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const auto boundary_mesh_interface_version =
      boundary_mesh_interface->get_version_function_handle()();

  if (func->version < boundary_mesh_interface_version) {
    const std::size_t tmp_buffer_size =
        num_accessible_geoms * func->cell_dof_count;
    func->d_dofs_stage->realloc_no_copy(tmp_buffer_size);
    boundary_mesh_interface->reverse_exchange_from_device(
        func->d_dofs->ptr, func->cell_dof_count, func->d_dofs_stage->ptr);
    func->version = boundary_mesh_interface_version;
  }

  REAL *k_buffer = func->d_dofs_stage->ptr;
  if (!null_sub_group) {
    auto *k_tree_root = d_tree_root;
    NESOASSERT(particle_sub_group->contains_ephemeral_dat(
                   Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
               "Boundary metadata not found on ParticleSubGroup.");
    NESOASSERT(
        (get_particle_group(particle_sub_group)->contains_dat(sym) &&
         (!is_ephemeral)) ||
            (particle_sub_group->contains_ephemeral_dat(sym) && is_ephemeral),
        "Destination particle dat not found.");

    ErrorPropagate ep(this->sycl_target);
    auto k_ep = ep.device_ptr();

    auto lambda_dispatch = [&](auto set_quantity) {
      particle_loop(
          "CartesianHMeshFunction::function_evaluate", particle_sub_group,
          [=](auto BOUNDARY_METADATA, auto SYM) {
            if (k_tree_root != nullptr) {
              const INT *index;
              bool found = false;
              found =
                  k_tree_root->get(BOUNDARY_METADATA.at_ephemeral(1), &index);
#ifndef NDEBUG
              NESO_KERNEL_ASSERT(found, k_ep);
#endif
              if (found) {
                set_quantity(SYM, component, k_buffer[*index]);
              }
            }
          },
          Access::read(Sym<INT>("NESO_PARTICLES_BOUNDARY_METADATA")),
          Access::write(sym))
          ->execute();
    };

    if (is_ephemeral) {
      lambda_dispatch([](auto &SYM, const int component, const REAL value) {
        SYM.at_ephemeral(component) = value;
      });
    } else {
      lambda_dispatch([](auto &SYM, const int component, const REAL value) {
        SYM.at(component) = value;
      });
    }

    NESOASSERT(!ep.get_flag(), "Failed to find index for hit geometry object.");
  }

  this->sycl_target->profile_map.end_region(r0);
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
  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection",
      "pre_integration_inner_particle_group");
  this->pre_integration_inner(particles);
  this->sycl_target->profile_map.end_region(r0);
  return;
}

void CartesianTrajectoryIntersection::pre_integration(
    std::shared_ptr<ParticleSubGroup> particles) {
  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection",
      "pre_integration_inner_particle_sub_group");
  this->pre_integration_inner(particles);
  this->sycl_target->profile_map.end_region(r0);
  return;
}

std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration(
    std::shared_ptr<ParticleGroup> particles) {
  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection",
      "post_integration_inner_particle_group");
  auto r = this->post_integration_inner(particles);
  this->sycl_target->profile_map.end_region(r0);
  return r;
}

std::map<int, ParticleSubGroupSharedPtr>
CartesianTrajectoryIntersection::post_integration(
    std::shared_ptr<ParticleSubGroup> particles) {
  auto r0 = this->sycl_target->profile_map.start_region(
      "CartesianTrajectoryIntersection",
      "post_integration_inner_particle_sub_group");
  auto r = this->post_integration_inner(particles);
  this->sycl_target->profile_map.end_region(r0);
  return r;
}

} // namespace NESO::Particles
