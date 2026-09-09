#ifndef _NESO_PARTICLES_GENERIC_FUNCTION_HPP_
#define _NESO_PARTICLES_GENERIC_FUNCTION_HPP_

#include "boundary/boundary_mesh_interface.hpp"
#include "device_buffers.hpp"
#include <string>

namespace NESO::Particles {

class GenericFunction;
typedef std::shared_ptr<GenericFunction> GenericFunctionSharedPtr;

/**
 * Initialise a surface function for projection.
 *
 * @param boundary_mesh_interface BoundaryMeshInterface used for
 * particle-surface intersections.
 * @param func Function created using the BoundaryMeshInterface.
 */
void prepare_surface_function_project_initialise(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func);

/**
 * Prepare a surface function for projection contributions.
 *
 * @param boundary_mesh_interface BoundaryMeshInterface used for
 * particle-surface intersections.
 * @param func Function created using the BoundaryMeshInterface.
 */
void prepare_surface_function_project_contribute(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func);

/**
 * Finalise a surface function projection.
 *
 * @param boundary_mesh_interface BoundaryMeshInterface used for
 * particle-surface intersections.
 * @param func Function created using the BoundaryMeshInterface.
 */
void prepare_surface_function_project_finalise(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func);

/**
 * Communicate DOFs and prepare the staging DOFs for use in function evaluation
 * at particle positions on a surface.
 *
 * @param boundary_mesh_interface BoundaryMeshInterface used for
 * particle-surface intersections.
 * @param func Function created using the BoundaryMeshInterface.
 */
void prepare_surface_function_evaluate(
    BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
    GenericFunctionSharedPtr func);

/**
 * Generic function type to represent finite element functions.
 */
class GenericFunction {

  friend void prepare_surface_function_evaluate(
      BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
      GenericFunctionSharedPtr func);

  friend void prepare_surface_function_project_finalise(
      BoundaryMeshInterfaceSharedPtr boundary_mesh_interface,
      GenericFunctionSharedPtr func);

protected:
  std::shared_ptr<BufferDevice<REAL>> d_dofs;
  std::shared_ptr<BufferDevice<REAL>> d_dofs_stage;
  std::int64_t version{0};
  void reset_version();
  int zeroed_stage_size{0};

  /**
   * Create a function on a mesh.
   *
   * @param sycl_target Compute device holding the DOFs.
   * @param ndim Number of spatial dimensions function exists in.
   * @param cell_count Number of locally owned cells on the mesh.
   * @param function_space Type of function to create.
   * @param polynomial_order Polynomial order of function to create.
   * @param mesh_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  GenericFunction(SYCLTargetSharedPtr sycl_target, const int ndim,
                  const int cell_count, const std::string function_space,
                  const int polynomial_order, const int mesh_group);

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
  int mesh_group{0};
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
   * @param mesh_group Label, e.g. boundary group, for subset of the mesh
   * this function is defined over.
   */
  GenericFunction(SYCLTargetSharedPtr sycl_target, const int ndim,
                  const std::vector<INT> &cells,
                  const std::string function_space, const int polynomial_order,
                  const int mesh_group);

  /**
   * @returns Device pointer to owned DOFs.
   */
  REAL *get_dofs_device_pointer();

  /**
   * Realloc the staging area to be at east a required size discarding contents.
   *
   * @param num_entries New number of entries.
   */
  void stage_realloc_no_copy(const int num_entries);

  /**
   * Realloc the staging area to be at east a required size keeping contents.
   *
   * @param num_entries New number of entries.
   */
  void stage_realloc(const int num_entries);

  /**
   * Reset the zeroing index to zero.
   */
  void stage_zero_reset();

  /**
   * Extend the zeroed elements in the stage from the last zeroing end to a
   * given number of entries.
   *
   * @param num_entries New number of entries.
   */
  void stage_extend_zero(const int num_entries);

  /**
   * @returns Device pointer to stage DOFs.
   */
  REAL *stage_get_dofs_device_pointer();

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

} // namespace NESO::Particles

#endif
