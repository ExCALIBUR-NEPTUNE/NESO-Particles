#include <neso_particles/generic_function.hpp>

namespace NESO::Particles {

void prepare_surface_function_project_initialise(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func) {
  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const std::size_t tmp_buffer_size =
      num_accessible_geoms * func->cell_dof_count;

  func->stage_realloc_no_copy(tmp_buffer_size);
  func->stage_zero_reset();
  func->stage_extend_zero(tmp_buffer_size);
  func->fill(0.0);
}

void prepare_surface_function_project_contribute(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func) {
  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const int k_num_accessible_geoms = num_accessible_geoms;
  const auto cell_dof_count = func->cell_dof_count;
  const int required_size = k_num_accessible_geoms * cell_dof_count;

  func->stage_realloc(required_size);
  func->stage_extend_zero(required_size);
}

void prepare_surface_function_project_finalise(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func) {
  boundary_mesh_interface->exchange_from_device(
      func->stage_get_dofs_device_pointer(), func->cell_dof_count,
      func->d_dofs->ptr);
  func->reset_version();
  NESOASSERT(func->version == 0, "Expected a version reset.");
}

void prepare_surface_function_evaluate(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func) {

  auto [d_tree_root, num_accessible_geoms] =
      boundary_mesh_interface->get_device_geom_id_to_seq();

  const auto boundary_mesh_interface_version =
      boundary_mesh_interface->get_version_function_handle()();

  // The function version is set to zero whenever the dofs are touched.
  if (func->version < boundary_mesh_interface_version) {
    const std::size_t tmp_buffer_size =
        num_accessible_geoms * func->cell_dof_count;
    func->stage_realloc_no_copy(tmp_buffer_size);
    boundary_mesh_interface->reverse_exchange_from_device(
        func->d_dofs->ptr, func->cell_dof_count,
        func->stage_get_dofs_device_pointer());
    func->version = boundary_mesh_interface_version;
  }
}

void GenericFunction::reset_version() { this->version = 0; }

GenericFunction::GenericFunction(SYCLTargetSharedPtr sycl_target,
                                 const int ndim, const int cell_count,
                                 const std::string function_space,
                                 const int polynomial_order,
                                 const int mesh_group)
    : sycl_target(sycl_target), ndim(ndim), cell_count(cell_count),
      function_space(function_space), polynomial_order(polynomial_order),
      mesh_group(mesh_group) {
  const int ndof_per_cell = std::pow(polynomial_order + 1, ndim);
  this->cell_dof_count = ndof_per_cell;
  this->local_dof_count = cell_count * ndof_per_cell;
  this->d_dofs =
      std::make_shared<BufferDevice<REAL>>(sycl_target, this->local_dof_count);
  this->d_dofs_stage = std::make_shared<BufferDevice<REAL>>(
      sycl_target, std::max(this->local_dof_count, 1));

  NESOASSERT(function_space == "DG",
             "Only currently implemented for DG0 functions.");
  NESOASSERT(polynomial_order == 0,
             "Only currently implemented for DG0 functions.");
  this->fill(0.0);
  this->reset_version();
}

GenericFunction::GenericFunction(SYCLTargetSharedPtr sycl_target,
                                 const int ndim, const std::vector<INT> &cells,
                                 const std::string function_space,
                                 const int polynomial_order,
                                 const int mesh_group)
    : GenericFunction(sycl_target, ndim, cells.size(), function_space,
                      polynomial_order, mesh_group) {
  this->cells = cells;
}

REAL *GenericFunction::get_dofs_device_pointer() { return this->d_dofs->ptr; }

void GenericFunction::stage_realloc_no_copy(const int num_entries) {
  if (num_entries > this->d_dofs_stage->size) {
    this->d_dofs_stage->realloc_no_copy(num_entries);
  }
}

void GenericFunction::stage_realloc(const int num_entries) {
  if (num_entries > this->d_dofs_stage->size) {
    this->d_dofs_stage->realloc(num_entries);
  }
}

void GenericFunction::stage_zero_reset() { this->zeroed_stage_size = 0; }

void GenericFunction::stage_extend_zero(const int num_entries) {

  const int end_old = this->zeroed_stage_size;
  const int end_new = num_entries;

  if (end_new <= end_old) {
    return;
  }

  NESOASSERT(end_new <= this->d_dofs_stage->size,
             "Cannot zero past the end of the allocated stage buffer. Was "
             "stage_realloc/stage_realloc_no_copy called?");

  REAL *RESTRICT k_base_ptr = this->d_dofs_stage->ptr;
  this->sycl_target->queue.fill(k_base_ptr + end_old, 0.0, end_new - end_old)
      .wait_and_throw();

  this->zeroed_stage_size = end_new;
}

REAL *GenericFunction::stage_get_dofs_device_pointer() {
  return this->d_dofs_stage->ptr;
}

void GenericFunction::write_vtkhdf(const std::string filename) {
  NESOASSERT(false, "Error not implemented.");
}

void GenericFunction::fill(const REAL value) {
  if (this->local_dof_count > 0) {
    this->sycl_target->queue
        .fill(static_cast<REAL *>(this->d_dofs->ptr), static_cast<REAL>(value),
              this->local_dof_count)
        .wait_and_throw();
  }
  this->reset_version();
  NESOASSERT(this->version == 0, "Expected a version reset.");
}

std::vector<REAL> GenericFunction::get_dofs() {
  std::vector<REAL> h_dofs(this->local_dof_count);
  if (this->local_dof_count > 0) {
    this->sycl_target->queue
        .memcpy(h_dofs.data(), this->d_dofs->ptr,
                this->local_dof_count * sizeof(REAL))
        .wait_and_throw();
  }
  return h_dofs;
}

void GenericFunction::set_dofs(std::vector<REAL> &h_dofs) {
  NESOASSERT(h_dofs.size() == this->local_dof_count,
             "h_dofs has the incorrect number of components.");
  if (this->local_dof_count > 0) {
    this->sycl_target->queue
        .memcpy(this->d_dofs->ptr, h_dofs.data(),
                this->local_dof_count * sizeof(REAL))
        .wait_and_throw();
  }
  this->reset_version();
  NESOASSERT(this->version == 0, "Expected a version reset.");
}

} // namespace NESO::Particles
