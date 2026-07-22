#ifndef _NESO_PARTICLES_GENERIC_FUNCTION_HPP_
#define _NESO_PARTICLES_GENERIC_FUNCTION_HPP_

#include "device_buffers.hpp"
#include <string>

namespace NESO::Particles {

/**
 * Generic function type to represent finite element functions.
 */
class GenericFunction {

protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  std::shared_ptr<BufferDevice<REAL>> d_dofs;
  std::shared_ptr<BufferDevice<REAL>> d_dofs_stage;
  std::int64_t version{0};
  void reset_version();

  /**
   * Create a function on a mesh.
   *
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cell_count Number of locally owned cells on the mesh.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param boundary_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  GenericFunction(SYCLTargetSharedPtr sycl_target, const int ndim,
                  const int cell_count, const std::string function_space,
                  const int polynomial_order, const int boundary_group);

public:
  /// Compute device holding the DOFs
  SYCLTargetSharedPtr sycl_target;
  /// The number of dimensions of the space this function is defined in.
  int ndim{0};
  /// The number of cells this function is defined over.
  int cell_count{0};
  /// The type of function, e.g. "DG".
  std::string function_space;
  /// The polynomial order of the function.
  int polynomial_order{0};
  /// The cells this function is defined over if there is redirection from the
  /// entity index to the cell index.
  std::vector<INT> cells;
  /// If this function corresponds to a boundary group then this entry records
  /// the boundary group.
  int boundary_group{0};
  /// Number of locally owned DOFs
  int local_dof_count{0};
  /// Number of DOFs per cell.
  int cell_dof_count{0};

  GenericFunction() = default;
  virtual ~GenericFunction() = default;

  /**
   * Create a function on a mesh on the passed entities.
   *
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cells Locally owned mesh entities to create function over.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param boundary_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  GenericFunction(SYCLTargetSharedPtr sycl_target, const int ndim,
                  const std::vector<INT> &cells,
                  const std::string function_space, const int polynomial_order,
                  const int boundary_group);

  /**
   * Write the function to a vtkhdf file. This function must be called
   * collectively on the communicator.
   *
   * @param filename Output file name which should have vtkhdf extension.
   */
  virtual void write_vtkhdf(const std::string filename);

  /**
   * Fill all the DOFs with a given value.
   *
   * @param value Value to assign to all DOFs.
   */
  virtual void fill(const REAL value);

  /**
   * @returns DOFs on host.
   */
  virtual std::vector<REAL> get_dofs();

  /**
   * Set the DOFs from a host vector. This function must be called collectively
   * on the communicator.
   *
   * @param h_dofs Host std::vector of length local_dof_count.
   */
  virtual void set_dofs(std::vector<REAL> &h_dofs);
};

typedef std::shared_ptr<GenericFunction> GenericFunctionSharedPtr;
} // namespace NESO::Particles

#endif
