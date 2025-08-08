#include <neso_particles/cartesian_mesh/cartesian_h_mesh_function.hpp>

namespace NESO::Particles {

CartesianHMeshFunction::CartesianHMeshFunction(
    CartesianHMeshSharedPtr mesh, SYCLTargetSharedPtr sycl_target,
    const int ndim, const int cell_count, const std::string function_space,
    const int polynomial_order, const int element_group)
    : mesh(mesh), sycl_target(sycl_target), ndim(ndim), cell_count(cell_count),
      function_space(function_space), polynomial_order(polynomial_order),
      element_group(element_group) {
  const int ndof_per_cell = std::pow(polynomial_order + 1, ndim);
  this->cell_dof_count = ndof_per_cell;
  this->local_dof_count = cell_count * ndof_per_cell;
  this->d_dofs =
      std::make_shared<BufferDevice<REAL>>(sycl_target, this->local_dof_count);
  NESOASSERT(ndim + 1 == mesh->get_ndim(),
             "Only currently implemented for boundary functions.");
  NESOASSERT(function_space == "DG",
             "Only currently implemented for DG0 functions.");
  NESOASSERT(polynomial_order == 0,
             "Only currently implemented for DG0 functions.");
  this->fill(0.0);
}

CartesianHMeshFunction::CartesianHMeshFunction(CartesianHMeshSharedPtr mesh,
                                               SYCLTargetSharedPtr sycl_target,
                                               const int ndim,
                                               const std::vector<INT> &cells,
                                               const std::string function_space,
                                               const int polynomial_order,
                                               const int element_group)
    : CartesianHMeshFunction(mesh, sycl_target, ndim, cells.size(),
                             function_space, polynomial_order, element_group) {
  this->cells = cells;
}

void CartesianHMeshFunction::write_vtkhdf(const std::string filename) {

  NESOASSERT(this->polynomial_order == 0, "Only implemented for DG0.");
  NESOASSERT(this->ndim + 1 == this->mesh->get_ndim(),
             "Only implemented for boundary cells.");

  std::vector<REAL> h_dofs(this->d_dofs->size);

  if (this->d_dofs->size > 0) {
    this->sycl_target->queue
        .memcpy(h_dofs.data(), this->d_dofs->ptr,
                this->d_dofs->size * sizeof(REAL))
        .wait_and_throw();
  }

  std::vector<VTK::UnstructuredCell> data;
  data.reserve(this->cells.size());

  std::size_t index = 0;
  for (auto &cx : this->cells) {
    auto vtkdata = this->mesh->get_vtk_face_cell_data(cx);
    vtkdata.cell_data["u"] = h_dofs.at(index++);
    data.push_back(vtkdata);
  }

  VTK::VTKHDF vtkhdf(filename, mesh->get_comm());
  vtkhdf.write(data, {}, {"u"});
  vtkhdf.close();
}

void CartesianHMeshFunction::fill(const REAL value) {
  if (this->local_dof_count > 0) {
    this->sycl_target->queue
        .fill(static_cast<REAL *>(this->d_dofs->ptr), static_cast<REAL>(value),
              this->local_dof_count)
        .wait_and_throw();
  }
}

std::vector<REAL> CartesianHMeshFunction::get_dofs() {
  std::vector<REAL> h_dofs(this->local_dof_count);
  if (this->local_dof_count > 0) {
    this->sycl_target->queue
        .memcpy(h_dofs.data(), this->d_dofs->ptr,
                this->local_dof_count * sizeof(REAL))
        .wait_and_throw();
  }
  return h_dofs;
}

void CartesianHMeshFunction::set_dofs(std::vector<REAL> &h_dofs) {
  NESOASSERT(h_dofs.size() == this->local_dof_count,
             "h_dofs has the incorrect number of components.");
  if (this->local_dof_count > 0) {
    this->sycl_target->queue
        .memcpy(this->d_dofs->ptr, h_dofs.data(),
                this->local_dof_count * sizeof(REAL))
        .wait_and_throw();
  }
}

} // namespace NESO::Particles
