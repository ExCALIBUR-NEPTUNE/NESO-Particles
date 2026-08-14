#ifdef NESO_PARTICLES_PETSC

#include <neso_particles/common_impl.hpp>
#include <neso_particles/communication/communication_utility.hpp>
#include <neso_particles/external_interfaces/petsc/boundary_interaction/boundary_interaction_common.hpp>

namespace NESO::Particles::PetscInterface {

namespace {
template <typename T>
inline void check_dat(ParticleGroupSharedPtr particle_group, Sym<T> sym,
                      const int ncomp) {
  if (!particle_group->contains_dat(sym)) {
    particle_group->add_particle_dat(sym, ncomp);
  } else {
    NESOASSERT(particle_group->get_dat(sym)->ncomp >= ncomp,
               "Requested dat with sym " + sym.name +
                   " exists already with an insufficient number of components");
  }
}
} // namespace

void BoundaryInteractionCommon::prepare_particle_group(
    ParticleGroupSharedPtr particle_group) {
  NESOASSERT(particle_group->sycl_target == this->sycl_target,
             "Missmatch of sycl targets.");
  const int ndim = this->mesh->get_ndim();
  check_dat(particle_group, this->previous_position_sym, ndim);
}

BoundaryInteractionCommon::BoundaryInteractionCommon(
    SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
    std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
    std::optional<Sym<REAL>> previous_position_sym)
    : sycl_target(sycl_target), mesh(mesh), boundary_groups(boundary_groups) {

  {
    std::set<int> contrib;
    for (auto &bx : boundary_groups) {
      contrib.insert(bx.first);
    }

    auto to_test = set_all_reduce_union(contrib, mesh->get_comm());
    NESOASSERT(to_test == contrib,
               "Missmatch in boundary group labels across ranks.");

    for (auto &bx : boundary_groups) {
      contrib.clear();
      for (auto &lx : bx.second) {
        contrib.insert(lx);
      }
      auto to_test = set_all_reduce_union(contrib, mesh->get_comm());
      NESOASSERT(to_test == contrib,
                 "Missmatch in boundary group DMPlex labels across ranks.");
    }
  }

  auto face_sets = mesh->dmh->get_face_sets();

  std::vector<INT> tmp_face_cells;

  for (auto &bx : boundary_groups) {
    tmp_face_cells.clear();
    NESOASSERT(bx.first >= 0, "Group id cannot be negative.");
    for (auto &lx : bx.second) {
      this->map_label_to_groups[lx] = bx.first;

      auto &labeled_face_points = face_sets[lx];
      for (const INT fx : labeled_face_points) {
        tmp_face_cells.push_back(fx);
      }
    }

    this->map_groups_boundary_interface[bx.first] =
        std::make_shared<BoundaryMeshInterface>(mesh->get_comm(), sycl_target,
                                                tmp_face_cells);
  }

  auto assign_sym = [=](auto &output_sym, auto &input_sym, auto default_sym) {
    if (input_sym != std::nullopt) {
      output_sym = input_sym.value();
    } else {
      output_sym = default_sym;
    }
  };
  assign_sym(this->previous_position_sym, previous_position_sym,
             Sym<REAL>("NESO_PARTICLES_DMPLEX_BOUNDARY_PREV_POS"));

  const int k_ndim = this->mesh->get_ndim();
  const int k_cell_count = this->mesh->get_cell_count();
  this->cdc_mh_min = std::make_shared<CellDatConst<int>>(
      this->sycl_target, k_cell_count, k_ndim, 1);
  this->cdc_mh_max = std::make_shared<CellDatConst<int>>(
      this->sycl_target, k_cell_count, k_ndim, 1);

  this->mesh_hierarchy_mapper = std::make_unique<MeshHierarchyMapper>(
      this->sycl_target, this->mesh->get_mesh_hierarchy());
  this->dh_max_box_size =
      std::make_unique<BufferDeviceHost<int>>(this->sycl_target, 1);

  const auto mesh_hierarchy_host_mapper =
      this->mesh_hierarchy_mapper->get_host_mapper();

  std::vector<int> h_cell_bounds = {0, 0, 0};
  for (int dimx = 0; dimx < k_ndim; dimx++) {
    const int max_possible_cell = mesh_hierarchy_host_mapper.dims[dimx] *
                                  mesh_hierarchy_host_mapper.ncells_dim_fine;
    h_cell_bounds.at(dimx) = max_possible_cell;
  }

  this->d_cell_bounds =
      std::make_unique<BufferDevice<int>>(this->sycl_target, h_cell_bounds);
  this->dh_mh_cells =
      std::make_unique<BufferDeviceHost<INT>>(this->sycl_target, 1024);
}

void BoundaryInteractionCommon::pre_integration(
    std::shared_ptr<ParticleGroup> particles) {
  auto particle_group = get_particle_group(particles);
  prepare_particle_group(particle_group);
  auto position_dat = particle_group->position_dat;
  const int k_ncomp = position_dat->ncomp;
  const int k_ndim = this->mesh->get_ndim();
  NESOASSERT(k_ncomp >= k_ndim,
             "Positions ncomp is smaller than the number of mesh dimensions.");

  particle_loop(
      "BoundaryInteractionCommon::pre_integration", particles,
      [=](auto P, auto PP) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          PP.at(dimx) = P.at(dimx);
        }
      },
      Access::read(position_dat->sym),
      Access::write(this->previous_position_sym))
      ->execute();
}

void BoundaryInteractionCommon::pre_integration(
    std::shared_ptr<ParticleSubGroup> particles) {
  auto particle_group = get_particle_group(particles);
  prepare_particle_group(particle_group);
  auto position_dat = particle_group->position_dat;
  const int k_ncomp = position_dat->ncomp;
  const int k_ndim = this->mesh->get_ndim();
  NESOASSERT(k_ncomp >= k_ndim,
             "Positions ncomp is smaller than the number of mesh dimensions.");

  particle_loop(
      "BoundaryInteractionCommon::pre_integration", particles,
      [=](auto P, auto PP) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          PP.at(dimx) = P.at(dimx);
        }
      },
      Access::read(position_dat->sym),
      Access::write(this->previous_position_sym))
      ->execute();
}

DMPlexFunctionSharedPtr
BoundaryInteractionCommon::create_function(const int group,
                                           const std::string function_space,
                                           const int polynomial_order) {
  auto r0 = this->sycl_target->profile_map.start_region(
      "BoundaryInteractionCommon", "create_function");

  NESOASSERT(boundary_groups.count(group),
             "Passed group is not a group ID known to this instance.");

  if (this->map_group_to_petsc_indices.count(group) == 0) {
    auto face_sets = this->mesh->dmh->get_face_sets();
    std::vector<INT> cells_tmp;
    for (const PetscInt labelx : this->boundary_groups.at(group)) {
      for (const PetscInt pointx : face_sets[labelx]) {
        cells_tmp.push_back(static_cast<INT>(pointx));
      }
    }
    this->map_group_to_petsc_indices[group] = cells_tmp;
  }

  const auto &cells = this->map_group_to_petsc_indices.at(group);

  auto func = std::make_shared<DMPlexFunction>(
      this->mesh, this->sycl_target, this->mesh->get_ndim() - 1, cells,
      function_space, polynomial_order, group);

  this->sycl_target->profile_map.end_region(r0);
  return func;
}

void BoundaryInteractionCommon::function_evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int component, const bool is_ephemeral,
    DMPlexFunctionSharedPtr func) {

  auto r0 = this->sycl_target->profile_map.start_region(
      "BoundaryInteractionCommon", "function_evaluate");

  NESOASSERT(func->function_space == "DG", "Only implemented for DG0.");
  NESOASSERT(func->polynomial_order == 0, "Only implemented for DG0.");

  const bool null_sub_group = particle_sub_group == nullptr;
  const int group = func->mesh_group;
  auto &boundary_mesh_interface = this->map_groups_boundary_interface.at(group);

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const auto boundary_mesh_interface_version =
      boundary_mesh_interface->get_version_function_handle()();

  // The function version is set to zero whenever the dofs are touched.
  if (func->version < boundary_mesh_interface_version) {
    const std::size_t tmp_buffer_size =
        num_accessible_geoms * func->cell_dof_count;
    func->d_dofs_stage->realloc_no_copy(tmp_buffer_size);
    boundary_mesh_interface->reverse_exchange_from_device(
        func->d_dofs->ptr, func->cell_dof_count, func->d_dofs_stage->ptr);
    func->version = boundary_mesh_interface_version;
  }

  if (!null_sub_group) {
    REAL const *const RESTRICT k_buffer = func->d_dofs_stage->ptr;
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
          "BoundaryInteractionCommon::function_evaluate", particle_sub_group,
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

} // namespace NESO::Particles::PetscInterface

#endif
