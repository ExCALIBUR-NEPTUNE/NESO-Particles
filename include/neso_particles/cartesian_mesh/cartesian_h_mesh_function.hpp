#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_FUNCTION_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_FUNCTION_HPP_

#include "../device_buffers.hpp"
#include "../generic_function.hpp"
#include "cartesian_h_mesh.hpp"
#include <string>

namespace NESO::Particles {

class CartesianTrajectoryIntersection;

/**
 * Generic function type to represent functions on CartesianHMesh.
 *
 */
class CartesianHMeshFunction : public GenericFunction {

  friend class CartesianTrajectoryIntersection;

protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

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
  CartesianHMeshFunction(CartesianHMeshSharedPtr mesh,
                         SYCLTargetSharedPtr sycl_target, const int ndim,
                         const int cell_count, const std::string function_space,
                         const int polynomial_order, const int boundary_group);

public:
  /// The mesh this function is defined on.
  CartesianHMeshSharedPtr mesh;

  CartesianHMeshFunction() = default;
  virtual ~CartesianHMeshFunction() = default;

  /**
   * Create a function on a mesh on the passed entities.
   *
   * @param mesh Host mesh to create function on.
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cells Locally owned mesh entities to create function over.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param boundary_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  CartesianHMeshFunction(CartesianHMeshSharedPtr mesh,
                         SYCLTargetSharedPtr sycl_target, const int ndim,
                         const std::vector<INT> &cells,
                         const std::string function_space,
                         const int polynomial_order, const int boundary_group);

  /**
   * Write the function to a vtkhdf file. This function must be called
   * collectively on the communicator.
   *
   * @param filename Output file name which should have vtkhdf extension.
   */
  virtual void write_vtkhdf(const std::string filename) override;
};

typedef std::shared_ptr<CartesianHMeshFunction> CartesianHMeshFunctionSharedPtr;
} // namespace NESO::Particles

#endif
