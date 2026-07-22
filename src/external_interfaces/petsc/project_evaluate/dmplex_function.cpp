#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/project_evaluate/dmplex_function.hpp>

namespace NESO::Particles::PetscInterface {

DMPlexFunction::DMPlexFunction(DMPlexInterfaceSharedPtr mesh,
                               SYCLTargetSharedPtr sycl_target, const int ndim,
                               const int cell_count,
                               const std::string function_space,
                               const int polynomial_order,
                               const int boundary_group)
    : GenericFunction(sycl_target, ndim, cell_count, function_space,
                      polynomial_order, boundary_group) {
  this->mesh = mesh;
}

DMPlexFunction::DMPlexFunction(DMPlexInterfaceSharedPtr mesh,
                               SYCLTargetSharedPtr sycl_target, const int ndim,
                               const std::vector<INT> &cells,
                               const std::string function_space,
                               const int polynomial_order,
                               const int boundary_group)
    : DMPlexFunction(mesh, sycl_target, ndim, cells.size(), function_space,
                     polynomial_order, boundary_group) {
  this->cells = cells;
}

} // namespace NESO::Particles::PetscInterface

#endif
