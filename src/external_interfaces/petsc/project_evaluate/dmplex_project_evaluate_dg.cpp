#ifdef NESO_PARTICLES_PETSC

#include <neso_particles/common_impl.hpp>
#include <neso_particles/external_interfaces/petsc/project_evaluate/dmplex_project_evaluate_dg.hpp>

namespace NESO::Particles::PetscInterface {

void DMPlexProjectEvaluateDG::check_setup() {
  if (this->qpm != nullptr) {
    NESOASSERT(this->qpm->points_added(),
               "QuadraturePointMapper needs points adding to it.");
  }
}

void DMPlexProjectEvaluateDG::check_ncomp(const int ncomp) {
  if (this->cdc_project->nrow < ncomp) {
    const int cell_count = this->mesh->get_cell_count();
    this->cdc_project = std::make_shared<CellDatConst<REAL>>(
        this->sycl_target, cell_count, ncomp, 1);
  }
}

std::vector<VTK::UnstructuredCell> DMPlexProjectEvaluateDG::get_vtk_data() {
  const int cell_count = this->mesh->get_cell_count();
  std::vector<VTK::UnstructuredCell> data =
      this->mesh->dmh->get_vtk_cell_data();
  const int ndim = mesh->get_ndim();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const auto cell_value = this->cdc_project->get_value(cellx, 0, 0);
    data.at(cellx).cell_data["value"] = cell_value;
  }
  return data;
}

DMPlexProjectEvaluateDG::DMPlexProjectEvaluateDG(
    ExternalCommon::QuadraturePointMapperSharedPtr qpm,
    std::string function_space, int polynomial_order)
    : DMPlexProjectEvaluateDG(
          std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
              qpm->domain->mesh),
          qpm->sycl_target, function_space, polynomial_order) {
  this->qpm = qpm;
  NESOASSERT(this->qpm != nullptr, "QuadraturePointMapper is nullptr");
}

DMPlexProjectEvaluateDG::DMPlexProjectEvaluateDG(
    DMPlexInterfaceSharedPtr mesh, SYCLTargetSharedPtr sycl_target,
    std::string function_space, int polynomial_order)
    : mesh(mesh), sycl_target(sycl_target), qpm(nullptr),
      function_space(function_space), polynomial_order(polynomial_order) {

  std::map<std::string, std::pair<int, int>> map_allowed;
  map_allowed["DG"] = {0, 0};

  NESOASSERT(map_allowed.count(function_space),
             "Only function space: " + function_space + " not recognised.");
  const int p_min = map_allowed.at(function_space).first;
  const int p_max = map_allowed.at(function_space).second;
  NESOASSERT(((p_min <= polynomial_order) && (polynomial_order <= p_max)),
             "Polynomial order " + std::to_string(polynomial_order) +
                 " outside of acceptable range [" + std::to_string(p_min) +
                 ", " + std::to_string(p_max) + "] for function space " +
                 function_space + ".");

  NESOASSERT(this->mesh != nullptr,
             "Mesh is not descendent from PetscInterface::DMPlexInterface");
  NESOASSERT(this->mesh->get_ndim() == 2, "Only implemented for 2D domains.");

  const int cell_count = this->mesh->get_cell_count();
  this->cdc_project =
      std::make_shared<CellDatConst<REAL>>(this->sycl_target, cell_count, 1, 1);
  this->cdc_volumes =
      std::make_shared<CellDatConst<REAL>>(this->sycl_target, cell_count, 1, 1);

  // For each cell record the volume.
  for (int cx = 0; cx < cell_count; cx++) {
    const auto volume = this->mesh->dmh->get_cell_volume(cx);
    this->cdc_volumes->set_value(cx, 0, 0, 1.0 / volume);
  }
}

void DMPlexProjectEvaluateDG::project(ParticleGroupSharedPtr particle_group,
                                      Sym<REAL> sym) {
  this->project_inner(particle_group, sym);
}

void DMPlexProjectEvaluateDG::project(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym) {
  this->project_inner(particle_sub_group, sym);
}

void DMPlexProjectEvaluateDG::evaluate(ParticleGroupSharedPtr particle_group,
                                       Sym<REAL> sym) {
  this->evaluate_inner(particle_group, sym);
}

void DMPlexProjectEvaluateDG::evaluate(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym) {
  this->evaluate_inner(particle_sub_group, sym);
}

void DMPlexProjectEvaluateDG::get_dofs(const int ncomp,
                                       std::vector<REAL> &dofs) {

  if (ncomp > 0) {
    NESOASSERT(this->cdc_project->nrow >= ncomp,
               "Requested more components than there are components in the "
               "internal representation.");

    auto h_dofs = this->cdc_project->get_all_cells();
    const int ncells = this->cdc_project->ncells;
    dofs.resize(ncells * ncomp);

    for (int cx = 0; cx < ncells; cx++) {
      for (int nx = 0; nx < ncomp; nx++) {
        dofs[cx * ncomp + nx] = h_dofs[cx]->at(nx, 0);
      }
    }
  } else {
    NESOWARN(false, "Number of components passed results in a no-op.");
    dofs.resize(0);
  }
}

void DMPlexProjectEvaluateDG::set_dofs(const int ncomp,
                                       const std::vector<REAL> &dofs) {
  if (ncomp > 0) {
    this->check_ncomp(ncomp);

    const int ncells = this->cdc_project->ncells;
    const int ncomp_dat = this->cdc_project->nrow;

    std::vector<CellData<REAL>> cell_data;
    cell_data.reserve(ncells);

    NESOASSERT(dofs.size() >= ncells * ncomp,
               "Passed DOF vector is too small for the number of cells and "
               "components.");

    for (int cellx = 0; cellx < ncells; cellx++) {
      cell_data.push_back(
          std::make_shared<CellDataT<REAL>>(this->sycl_target, ncomp_dat, 1));
    }

    for (int cx = 0; cx < ncells; cx++) {
      for (int nx = 0; nx < ncomp; nx++) {
        cell_data[cx]->at(nx, 0) = dofs[cx * ncomp + nx];
      }
      for (int nx = ncomp; nx < ncomp_dat; nx++) {
        cell_data[cx]->at(nx, 0) = 0.0;
      }
    }

    this->cdc_project->set_all_cells(cell_data);

  } else {
    NESOWARN(false, "Number of components passed results in a no-op.");
  }
}

} // namespace NESO::Particles::PetscInterface
#endif
