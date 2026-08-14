#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_PROJECT_EVALUATE_DMPLEX_FUNCTION_HPP_
#define _NESO_PARTICLES_EXTERNAL_PETSC_PROJECT_EVALUATE_DMPLEX_FUNCTION_HPP_

#include "../../../generic_function.hpp"
#include "../dmplex_interface.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Generic type for functions on DMPlex volumes and surfaces.
 */
class DMPlexFunction : public GenericFunction {
protected:
  std::vector<VTK::UnstructuredCell> vtk_data;
  std::vector<INT> cells_local;

  /**
   * Create a function on a mesh.
   *
   * @param mesh Host mesh to create function on.
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cell_count Number of locally owned cells on the mesh.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param boundary_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  DMPlexFunction(DMPlexInterfaceSharedPtr mesh, SYCLTargetSharedPtr sycl_target,
                 const int ndim, const int cell_count,
                 const std::string function_space, const int polynomial_order,
                 const int boundary_group);

public:
  DMPlexFunction() = default;
  virtual ~DMPlexFunction() = default;

  // The mesh the function is defined on.
  DMPlexInterfaceSharedPtr mesh;

  /**
   * Create a function on a mesh on the passed entities.
   *
   * @param mesh Host mesh to create function on.
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cells_local Locally owned mesh entities to create function over.
   * These must be point indices not cell indices.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param boundary_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  DMPlexFunction(DMPlexInterfaceSharedPtr mesh, SYCLTargetSharedPtr sycl_target,
                 const int ndim, const std::vector<INT> &cells_local,
                 const std::string function_space, const int polynomial_order,
                 const int boundary_group);

  /**
   * Write the function to a vtkhdf file. This function must be called
   * collectively on the communicator.
   *
   * @param filename Output file name which should have vtkhdf extension.
   */
  virtual void write_vtkhdf(const std::string filename) override;
};

using DMPlexFunctionSharedPtr = std::shared_ptr<DMPlexFunction>;

} // namespace NESO::Particles::PetscInterface

#endif
