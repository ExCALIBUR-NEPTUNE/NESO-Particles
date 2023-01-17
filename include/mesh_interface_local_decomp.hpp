#ifndef _NESO_PARTICLES_MESH_INTERFACE_LOCAL_DECOMP
#define _NESO_PARTICLES_MESH_INTERFACE_LOCAL_DECOMP
#include "mesh_interface.hpp"
#include "typedefs.hpp"
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <set>
#include <vector>

namespace NESO::Particles {

/**
 * A simple HMesh type that does not attempt to build any global mapping data
 * structures.
 */
class LocalDecompositionHMesh : public HMesh {
private:
public:
  /// Number of dimensions (physical).
  int ndim;
  /// Subdivision order of MeshHierarchy.
  int subdivision_order;
  /// MPI Communicator used.
  MPI_Comm comm;
  /// Underlying MeshHierarchy instance.
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  /// Number of cells, i.e. Number of Nektar++ elements on this rank.
  int cell_count;
  /// Global origin of domain.
  std::array<double, 3> global_origin;
  /// Global extents of global bounding box.
  std::array<double, 3> global_extents;
  /// Vector of nearby ranks which local exchange patterns can be setup with.
  std::vector<int> neighbour_ranks;

  ~LocalDecompositionHMesh() {}

  /**
   * Create a new instance.
   *
   * @param ndim Number of dimensions.
   * @param origin Origin to use.
   * @param extents Cartesian extents of the domain.
   * @param cell_count Number of cells in the mesh.
   * @param comm MPI communicator to use.
   */
  LocalDecompositionHMesh(const int ndim, std::vector<double> origin,
                          std::vector<double> extents, const int cell_count,
                          MPI_Comm comm = MPI_COMM_WORLD)
      : comm(comm), ndim(ndim), cell_count(cell_count) {

    std::vector<int> dims(this->ndim);
    for (int dimx = 0; dimx < 3; dimx++) {
      this->global_origin[dimx] = origin[dimx];
      this->global_extents[dimx] = extents[dimx];
      dims[dimx] = 1;
    }

    // create the mesh hierarchy
    this->mesh_hierarchy = std::make_shared<MeshHierarchy>(
        this->comm, this->ndim, dims, origin, 1.0, 0);
  };

  /**
   * Get the MPI communicator of the mesh.
   *
   * @returns MPI communicator.
   */
  inline MPI_Comm get_comm() { return this->comm; };
  /**
   *  Get the number of dimensions of the mesh.
   *
   *  @returns Number of mesh dimensions.
   */
  inline int get_ndim() { return this->ndim; };
  /**
   *  Get the Mesh dimensions.
   *
   *  @returns Mesh dimensions.
   */
  inline std::vector<int> &get_dims() { return this->mesh_hierarchy->dims; };
  /**
   * Get the subdivision order of the mesh.
   *
   * @returns Subdivision order.
   */
  inline int get_subdivision_order() {
    return this->mesh_hierarchy->subdivision_order;
  };
  /**
   * Get the total number of cells in the mesh on this MPI rank, i.e. the
   * number of Nektar++ elements on this MPI rank.
   *
   * @returns Total number of mesh cells on this MPI rank.
   */
  inline int get_cell_count() { return this->cell_count; };
  /**
   * Get the mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy coarse cell width.
   */
  inline double get_cell_width_coarse() {
    return this->mesh_hierarchy->cell_width_coarse;
  };
  /**
   * Get the mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy fine cell width.
   */
  inline double get_cell_width_fine() {
    return this->mesh_hierarchy->cell_width_fine;
  };
  /**
   * Get the inverse mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse coarse cell width.
   */
  inline double get_inverse_cell_width_coarse() {
    return this->mesh_hierarchy->inverse_cell_width_coarse;
  };
  /**
   * Get the inverse mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse fine cell width.
   */
  inline double get_inverse_cell_width_fine() {
    return this->mesh_hierarchy->inverse_cell_width_fine;
  };
  /**
   *  Get the global number of coarse cells.
   *
   *  @returns Global number of coarse cells.
   */
  inline int get_ncells_coarse() {
    return static_cast<int>(this->mesh_hierarchy->ncells_coarse);
  };
  /**
   *  Get the number of fine cells per coarse cell.
   *
   *  @returns Number of fine cells per coarse cell.
   */
  inline int get_ncells_fine() {
    return static_cast<int>(this->mesh_hierarchy->ncells_fine);
  };
  /**
   * Get the MeshHierarchy instance placed over the mesh.
   *
   * @returns MeshHierarchy placed over the mesh.
   */
  inline std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() {
    return this->mesh_hierarchy;
  };
  /**
   *  Free the mesh and associated communicators.
   */
  inline void free() { this->mesh_hierarchy->free(); }
  /**
   *  Get a std::vector of MPI ranks which should be used to setup local
   *  communication patterns.
   *
   *  @returns std::vector of MPI ranks.
   */
  inline std::vector<int> &get_local_communication_neighbours() {
    return this->neighbour_ranks;
  };
  /**
   *  Get a point in the domain that should be in, or at least close to, the
   *  sub-domain on this MPI process. Useful for parallel initialisation.
   *
   *  @param point Pointer to array of size equal to at least the number of mesh
   * dimensions.
   */
  inline void get_point_in_subdomain(double *point) {
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      point[dimx] = this->global_origin[dimx];
    }
  };
};

} // namespace NESO::Particles

#endif
