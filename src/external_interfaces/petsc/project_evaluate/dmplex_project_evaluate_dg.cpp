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
  auto r0 = sycl_target->profile_map.start_region("DMPlexProjectEvaluateDG",
                                                  "get_vtk_data");

  const int cell_count = this->mesh->get_cell_count();
  std::vector<VTK::UnstructuredCell> data =
      this->mesh->dmh->get_vtk_cell_data();
  const int ndim = mesh->get_ndim();
  const int ncomp = this->ncomp_active;
  const int stride = this->cdc_project->nrow;

  auto h_data =
      get_resource<BufferHost<REAL>, ResourceStackInterfaceBufferHost<REAL>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferHost<REAL>{},
          sycl_target);
  h_data->realloc_no_copy(cell_count * stride);

  this->sycl_target->queue
      .memcpy(h_data->ptr, this->cdc_project->device_ptr(),
              cell_count * stride * sizeof(REAL))
      .wait_and_throw();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int cx = 0; cx < ncomp; cx++) {
      const REAL cell_value = h_data->ptr[cellx * stride + cx];
      data.at(cellx).cell_data["value_" + std::to_string(cx)] = cell_value;
    }
  }

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferHost<REAL>{}, h_data);

  sycl_target->profile_map.end_region(r0);
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

  auto r0 = sycl_target->profile_map.start_region("DMPlexProjectEvaluateDG",
                                                  "get_dofs");

  if (ncomp > 0) {

    const int stride = this->cdc_project->nrow;
    NESOASSERT(stride >= ncomp,
               "Requested more components than there are components in the "
               "internal representation.");

    const int ncells = this->cdc_project->ncells;
    dofs.resize(ncells * ncomp);

    auto d_data = get_resource<BufferDevice<REAL>,
                               ResourceStackInterfaceBufferDevice<REAL>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
        sycl_target);
    d_data->realloc_no_copy(ncells * ncomp);
    REAL *RESTRICT k_data = d_data->ptr;

    REAL const *const RESTRICT k_src = this->cdc_project->device_ptr();

    auto e0 = this->sycl_target->queue.parallel_for(
        sycl::range<2>(ncells, ncomp), [=](sycl::item<2> idx) {
          const std::size_t cellx = idx.get_id(0);
          const std::size_t component = idx.get_id(1);
          k_data[cellx * ncomp + component] = k_src[cellx * stride + component];
        });

    this->sycl_target->queue
        .memcpy(dofs.data(), k_data, ncells * ncomp * sizeof(REAL), e0)
        .wait_and_throw();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<REAL>{}, d_data);
  } else {
    NESOWARN(false, "Number of components passed results in a no-op.");
    dofs.resize(0);
  }

  sycl_target->profile_map.end_region(r0);
}

void DMPlexProjectEvaluateDG::set_dofs(const int ncomp,
                                       const std::vector<REAL> &dofs) {

  auto r0 = sycl_target->profile_map.start_region("DMPlexProjectEvaluateDG",
                                                  "set_dofs");
  if (ncomp > 0) {
    this->check_ncomp(ncomp);
    this->ncomp_active = ncomp;

    const int ncells = this->cdc_project->ncells;
    const int stride = this->cdc_project->nrow;

    NESOASSERT(dofs.size() >= ncells * ncomp,
               "Passed DOF vector is too small for the number of cells and "
               "components.");

    auto d_data = get_resource<BufferDevice<REAL>,
                               ResourceStackInterfaceBufferDevice<REAL>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
        sycl_target);
    d_data->realloc_no_copy(ncells * ncomp);
    REAL *RESTRICT k_data = d_data->ptr;

    auto e0 = this->sycl_target->queue.memcpy(k_data, dofs.data(),
                                              ncells * ncomp * sizeof(REAL));

    REAL *RESTRICT k_dst = this->cdc_project->device_ptr();

    {

      REAL const *const RESTRICT k_src = k_data;
      this->sycl_target->queue
          .parallel_for(sycl::range<2>(ncells, ncomp), e0,
                        [=](sycl::item<2> idx) {
                          const std::size_t cellx = idx.get_id(0);
                          const std::size_t component = idx.get_id(1);
                          k_dst[cellx * stride + component] =
                              k_src[cellx * ncomp + component];
                        })
          .wait_and_throw();
    }

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<REAL>{}, d_data);

  } else {
    NESOWARN(false, "Number of components passed results in a no-op.");
  }

  sycl_target->profile_map.end_region(r0);
}

} // namespace NESO::Particles::PetscInterface
#endif
