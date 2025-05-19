#ifdef NESO_PARTICLES_PETSC

#include <neso_particles/common_impl.hpp>
#include <neso_particles/external_interfaces/petsc/boundary_interaction/boundary_interaction_common.hpp>

namespace NESO::Particles::PetscInterface {

BoundaryInteractionCommon::BoundaryInteractionCommon(
    SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
    std::map<PetscInt, std::vector<PetscInt>> &boundary_groups,
    std::optional<Sym<REAL>> previous_position_sym,
    std::optional<Sym<REAL>> boundary_position_sym,
    std::optional<Sym<INT>> boundary_label_sym)
    : sycl_target(sycl_target), mesh(mesh), boundary_groups(boundary_groups) {

  for (auto &bx : boundary_groups) {
    NESOASSERT(bx.first >= 0, "Group id cannot be negative.");
    for (auto &lx : bx.second) {
      this->map_label_to_groups[lx] = bx.first;
    }
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
  assign_sym(this->boundary_position_sym, boundary_position_sym,
             Sym<REAL>("NESO_PARTICLES_DMPLEX_BOUNDARY_POS"));
  assign_sym(this->boundary_label_sym, boundary_label_sym,
             Sym<INT>("NESO_PARTICLES_DMPLEX_BOUNDARY_LABEL"));

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
  prepare_particle_group(particles);
  auto particle_group = this->get_particle_group(particles);
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
  prepare_particle_group(particles);
  auto particle_group = this->get_particle_group(particles);
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

} // namespace NESO::Particles::PetscInterface

#endif
