#include <neso_particles/cartesian_mesh/cartesian_h_mesh_function.hpp>

namespace NESO::Particles {

CartesianHMeshFunction::CartesianHMeshFunction(
    CartesianHMeshSharedPtr mesh, SYCLTargetSharedPtr sycl_target,
    const int ndim, const int cell_count, const std::string function_space,
    const int polynomial_order, const int boundary_group)
    : GenericFunction(sycl_target, ndim, cell_count, function_space,
                      polynomial_order, boundary_group) {
  this->mesh = mesh;
  this->boundary_group = boundary_group;
}

CartesianHMeshFunction::CartesianHMeshFunction(CartesianHMeshSharedPtr mesh,
                                               SYCLTargetSharedPtr sycl_target,
                                               const int ndim,
                                               const std::vector<INT> &cells,
                                               const std::string function_space,
                                               const int polynomial_order,
                                               const int boundary_group)
    : CartesianHMeshFunction(mesh, sycl_target, ndim, cells.size(),
                             function_space, polynomial_order, boundary_group) {
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

} // namespace NESO::Particles
