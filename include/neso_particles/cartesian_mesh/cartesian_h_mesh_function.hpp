#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_FUNCTION_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_FUNCTION_HPP_

#include "cartesian_h_mesh.hpp"
#include <string>

namespace NESO::Particles {

/**
 * Generic function type to represent functions on CartesianHMesh.
 */
class CartesianHMeshFunction {
protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  CartesianHMeshSharedPtr mesh;

public:
  /// The number of dimensions of the space this function is defined in.
  int ndim{0};
  /// The number of cells this function is defined over.
  int cell_count{0};
  /// The cells this function is defined over if there is redirection from the
  /// entity index to the cell index.
  std::vector<INT> cells;

  CartesianHMeshFunction() = default;
  ~CartesianHMeshFunction() = default;

  /**
   * Create a function on a mesh.
   *
   * @param mesh Host mesh to create function on.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cell_count Number of locally owned cells on the mesh.
   * @param function_family Type of function to create.
   * @param function_order Polynomial order of function to create.
   */
  CartesianHMeshFunction(CartesianHMeshSharedPtr mesh, const int ndim,
                         const int cell_count,
                         const std::string function_family,
                         const int function_order)
      : mesh(mesh), ndim(ndim), cell_count(cell_count) {}

  /**
   * Create a function on a mesh on the passed entities.
   *
   * @param mesh Host mesh to create function on.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cells Locally owned mesh entities to create function over.
   * @param function_family Type of function to create.
   * @param function_order Polynomial order of function to create.
   */

  CartesianHMeshFunction(CartesianHMeshSharedPtr mesh, const int ndim,
                         const std::vector<INT> &cells,
                         const std::string function_family,
                         const int function_order)
      : CartesianHMeshFunction(mesh, ndim, cells.size(), function_family,
                               function_order) {
    this->cells = cells;
  }
};

} // namespace NESO::Particles

#endif
