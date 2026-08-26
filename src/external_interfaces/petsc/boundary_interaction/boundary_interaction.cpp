#ifdef NESO_PARTICLES_PETSC

#include <neso_particles/external_interfaces/petsc/boundary_interaction/boundary_interaction.hpp>

namespace NESO::Particles::PetscInterface {

std::shared_ptr<BoundaryInteractionCommon> create_boundary_interaction(
    SYCLTargetSharedPtr sycl_target, DMPlexInterfaceSharedPtr mesh,
    std::map<PetscInt, std::vector<PetscInt>> &boundary_groups, const REAL tol,
    std::optional<Sym<REAL>> previous_position_sym) {

  if (mesh->get_ndim() == 2) {
    auto ptr = std::dynamic_pointer_cast<BoundaryInteractionCommon>(
        std::make_shared<BoundaryInteraction2D>(
            sycl_target, mesh, boundary_groups, tol, previous_position_sym));
    NESOASSERT(ptr != nullptr,
               "Failed to cast pointer to BoundaryInteractionCommon");
    return ptr;

  } else if (mesh->get_ndim() == 3) {
    auto ptr = std::dynamic_pointer_cast<BoundaryInteractionCommon>(
        std::make_shared<BoundaryInteraction3D>(
            sycl_target, mesh, boundary_groups, tol, previous_position_sym));
    NESOASSERT(ptr != nullptr,
               "Failed to cast pointer to BoundaryInteractionCommon");
    return ptr;

  } else {
    NESOASSERT(false, "Bad mesh dimension.");
    return nullptr;
  }
}

} // namespace NESO::Particles::PetscInterface

#endif
