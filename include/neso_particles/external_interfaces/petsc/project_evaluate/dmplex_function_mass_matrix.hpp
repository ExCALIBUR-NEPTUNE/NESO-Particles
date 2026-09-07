#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_PROJECT_EVALUATE_DMPLEX_FUNCTION_MASS_MATRIX_HPP_
#define _NESO_PARTICLES_EXTERNAL_PETSC_PROJECT_EVALUATE_DMPLEX_FUNCTION_MASS_MATRIX_HPP_

#include "dmplex_function.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Implementation of mass matrix solves for DMPlexFunctions.
 */
class DMPlexFunctionMassMatrix {
protected:
  SYCLTargetSharedPtr sycl_target;
  int mesh_group{0};
  int num_cells{0};
  std::shared_ptr<BufferDevice<REAL>> d_inverse_mass_matrix;

public:
  /**
   * Create instance to solve mass matrix systems for a given function space.
   *
   * @param mesh Host mesh to create function on.
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cells_local Locally owned mesh entities to create function over.
   * These must be point indices not cell indices.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param mesh_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  DMPlexFunctionMassMatrix(DMPlexInterfaceSharedPtr mesh,
                           SYCLTargetSharedPtr sycl_target, const int ndim,
                           const std::vector<INT> &cells_local,
                           const std::string function_space,
                           const int polynomial_order, const int mesh_group);

  /**
   * Solve mass matrix system. Assume that the DOFs in the passed function are
   * actually the RHS of the system
   *
   *  M x = b.
   *
   *  @param[in, out] func Function to solve mass matrix system for.
   */
  void solve(DMPlexFunctionSharedPtr func);
};
} // namespace NESO::Particles::PetscInterface

#endif
