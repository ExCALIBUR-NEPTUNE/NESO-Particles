#ifndef _NESO_PARTICLES_MESH_INTERFACE
#define _NESO_PARTICLES_MESH_INTERFACE
#include "mesh_hierarchy.hpp"
#include "typedefs.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <set>
#include <vector>

namespace NESO::Particles {

/**
 *  Abstract base class for mesh types over which a MeshHierarchy is placed.
 */
class HMesh {

public:
  /**
   * Get the MPI communicator of the mesh.
   *
   * @returns MPI communicator.
   */
  virtual inline MPI_Comm get_comm() = 0;
  /**
   *  Get the number of dimensions of the mesh.
   *
   *  @returns Number of mesh dimensions.
   */
  virtual inline int get_ndim() = 0;
  /**
   *  Get the Mesh dimensions.
   *
   *  @returns Mesh dimensions.
   */
  virtual inline std::vector<int> &get_dims() = 0;
  /**
   * Get the subdivision order of the mesh.
   *
   * @returns Subdivision order.
   */
  virtual inline int get_subdivision_order() = 0;
  /**
   * Get the total number of cells in the mesh.
   *
   * @returns Total number of mesh cells.
   */
  virtual inline int get_cell_count() = 0;
  /**
   * Get the mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy coarse cell width.
   */
  virtual inline double get_cell_width_coarse() = 0;
  /**
   * Get the mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy fine cell width.
   */
  virtual inline double get_cell_width_fine() = 0;
  /**
   * Get the inverse mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse coarse cell width.
   */
  virtual inline double get_inverse_cell_width_coarse() = 0;
  /**
   * Get the inverse mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse fine cell width.
   */
  virtual inline double get_inverse_cell_width_fine() = 0;
  /**
   *  Get the global number of coarse cells.
   *
   *  @returns Global number of coarse cells.
   */
  virtual inline int get_ncells_coarse() = 0;
  /**
   *  Get the number of fine cells per coarse cell.
   *
   *  @returns Number of fine cells per coarse cell.
   */
  virtual inline int get_ncells_fine() = 0;
  /**
   * Get the MeshHierarchy instance placed over the mesh.
   *
   * @returns MeshHierarchy placed over the mesh.
   */
  virtual inline std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() = 0;
  /**
   *  Free the mesh and associated communicators.
   */
  virtual inline void free() = 0;
  /**
   *  Get a std::vector of MPI ranks which should be used to setup local
   *  communication patterns.
   *
   *  @returns std::vector of MPI ranks.
   */
  virtual inline std::vector<int> &get_local_communication_neighbours() = 0;
  /**
   *  Get a point in the domain that should be in, or at least close to, the
   *  sub-domain on this MPI process. Useful for parallel initialisation.
   *
   *  @param point Pointer to array of size equal to at least the number of mesh
   * dimensions.
   */
  virtual inline void get_point_in_subdomain(double *point) = 0;
};

typedef std::shared_ptr<HMesh> HMeshSharedPtr;

} // namespace NESO::Particles

#endif
