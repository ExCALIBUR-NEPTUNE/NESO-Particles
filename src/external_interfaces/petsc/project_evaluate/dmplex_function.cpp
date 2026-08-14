#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/project_evaluate/dmplex_function.hpp>

namespace NESO::Particles::PetscInterface {

namespace {
std::vector<INT> get_all_cell_point_indices(DMPlexInterfaceSharedPtr mesh) {
  const int cell_count = mesh->get_cell_count();
  std::vector<INT> points(cell_count);
  mesh->dmh->get_cell_petsc_indices(points);
  return points;
}
} // namespace

DMPlexFunction::DMPlexFunction(DMPlexInterfaceSharedPtr mesh,
                               SYCLTargetSharedPtr sycl_target, const int ndim,
                               const int cell_count,
                               const std::string function_space,
                               const int polynomial_order,
                               const int boundary_group)
    :

      DMPlexFunction(mesh, sycl_target, ndim, get_all_cell_point_indices(mesh),
                     function_space, polynomial_order, boundary_group) {}

DMPlexFunction::DMPlexFunction(DMPlexInterfaceSharedPtr mesh,
                               SYCLTargetSharedPtr sycl_target, const int ndim,
                               const std::vector<INT> &cells_local,
                               const std::string function_space,
                               const int polynomial_order,
                               const int boundary_group)
    :

      GenericFunction(sycl_target, ndim, cells_local.size(), function_space,
                      polynomial_order, boundary_group)

{
  this->mesh = mesh;
  this->cells_local = cells_local;
  this->cells.clear();
  this->cells.reserve(this->cells_local.size());
  for (auto cx : this->cells_local) {
    this->cells.push_back(this->mesh->dmh->get_point_global_index(cx));
  }
}

void DMPlexFunction::write_vtkhdf(const std::string filename) {

  if (this->vtk_data.size() != this->cells_local.size()) {
    this->vtk_data.reserve(this->cells_local.size());
    for (const INT pointx : this->cells_local) {
      this->vtk_data.push_back(this->mesh->dmh->get_vtk_point_data(pointx));
    }
  }
  NESOASSERT(this->polynomial_order == 0, "Only implemented for DG0.");
  NESOASSERT(this->vtk_data.size() == cell_count, "Size missmatch.");

  auto h_dofs = this->get_dofs();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    this->vtk_data[cellx].cell_data["u"] = h_dofs[cellx];
  }

  VTK::VTKHDF vtkhdf(filename, mesh->get_comm());
  vtkhdf.write(this->vtk_data, {}, {"u"});
  vtkhdf.close();
}

} // namespace NESO::Particles::PetscInterface

#endif
