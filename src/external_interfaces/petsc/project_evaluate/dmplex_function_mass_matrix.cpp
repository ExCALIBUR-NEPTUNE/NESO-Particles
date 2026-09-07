#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/project_evaluate/dmplex_function_mass_matrix.hpp>

namespace NESO::Particles::PetscInterface {

DMPlexFunctionMassMatrix::DMPlexFunctionMassMatrix(
    DMPlexInterfaceSharedPtr mesh, SYCLTargetSharedPtr sycl_target,
    const int ndim, const std::vector<INT> &cells_local,
    const std::string function_space, const int polynomial_order,
    const int mesh_group)
    : sycl_target(sycl_target), mesh_group(mesh_group) {
  NESOASSERT(function_space == "DG", "Only implemented for DG0");
  NESOASSERT(polynomial_order == 0, "Only implemented for DG0");
  NESOASSERT((0 < ndim) && (ndim < 4),
             "Only implemented in 1 to 3 dimensions.");

  const int num_points = cells_local.size();
  this->num_cells = num_points;

  if (this->num_cells > 0) {

    std::vector<REAL> h_inverse_mass_matrix(cells_local.size());
    for (int pointx = 0; pointx < num_points; pointx++) {
      const PetscInt point_index = cells_local[pointx];
      const REAL volume = mesh->dmh->get_point_volume(point_index);
      const REAL inverse_volume = 1.0 / volume;
      h_inverse_mass_matrix[pointx] = inverse_volume;
    }

    // DG0 mass matrix is a diagonal matrix of the cell volume.
    this->d_inverse_mass_matrix = std::make_shared<BufferDevice<REAL>>(
        this->sycl_target, h_inverse_mass_matrix);
  }
}

void DMPlexFunctionMassMatrix::solve(DMPlexFunctionSharedPtr func) {

  if ((func == nullptr) || (this->num_cells == 0)) {
    return;
  }

  NESOASSERT(func->cell_count == this->num_cells, "Num cells missmatch.");
  NESOASSERT(func->mesh_group == this->mesh_group, "Mesh group missmatch.");

  REAL const *const RESTRICT k_inverse_mass_matrix =
      this->d_inverse_mass_matrix->ptr;
  REAL *RESTRICT k_dofs = func->get_dofs_device_pointer();

  this->sycl_target->queue
      .parallel_for(sycl::range<1>(this->num_cells),
                    [=](sycl::item<1> idx) {
                      const auto ix = idx.get_id(0);
                      k_dofs[ix] = k_inverse_mass_matrix[ix] * k_dofs[ix];
                    })
      .wait_and_throw();
}

} // namespace NESO::Particles::PetscInterface
#endif
