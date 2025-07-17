#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_H_MESH_HPP_
#include "../mesh_interface.hpp"

#include "../external_interfaces/vtk/vtk.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <set>
#include <unordered_map>
#include <vector>

namespace NESO::Particles {

/**
 * Example mesh that duplicates a MeshHierarchy as a HMesh for examples and
 * testing.
 */
class CartesianHMesh : public HMesh {
private:
  int cell_count;
  MPI_Comm comm_cart;
  int periods[3] = {1, 1, 1};
  int coords[3] = {0, 0, 0};
  int mpi_dims[3] = {0, 0, 0};
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  bool allocated = false;
  std::vector<int> neighbour_ranks;

  int num_face_geoms{0};
  std::array<int, 6> face_strides0{0, 0, 0, 0, 0, 0};
  std::array<int, 6> face_strides1{0, 0, 0, 0, 0, 0};
  std::array<int, 6> num_geoms_per_face{0, 0, 0, 0, 0, 0};
  std::array<int, 6> num_geoms_per_face_incscan{0, 0, 0, 0, 0, 0};
  std::array<int, 6> num_geoms_per_face_exscan{0, 0, 0, 0, 0, 0};
  std::unordered_map<int, int> cache_map_face_id_to_rank;
  std::vector<INT> owned_face_indices;

  void compute_owned_face_indices();
  MPI_Comm comm_faces{MPI_COMM_NULL};

public:
  /// Disable (implicit) copies.
  CartesianHMesh(const CartesianHMesh &st) = delete;
  /// Disable (implicit) copies.
  CartesianHMesh &operator=(CartesianHMesh const &a) = delete;

  virtual ~CartesianHMesh() = default;

  /// Holds the first cell this rank owns in each dimension.
  int cell_starts[3] = {0, 0, 0};
  /// Holds the last cell+1 this ranks owns in each dimension.
  int cell_ends[3] = {1, 1, 1};
  /// Global number of cells in each dimension.
  std::vector<int> cell_counts = {0, 0, 0};
  /// Local number of cells in each dimension.
  int cell_counts_local[3] = {0, 0, 0};
  /// Global extents of the mesh.
  double global_extents[3] = {0.0, 0.0, 0.0};
  /// Width of the stencil used to determine which MPI ranks are neighbours.
  int stencil_width;
  /// Number of dimensions of the mesh.
  const int ndim;
  /// Vector holding the number of coarse cells in each dimension.
  std::vector<int> &dims;
  /// Subdivision order to determine number of fine cells per coarse cell.
  const int subdivision_order;
  /// Width of coarse cells, uniform in all dimensions.
  const double cell_width_coarse;
  /// Width of fine cells, uniform in all dimensions.
  const double cell_width_fine;
  /// Inverse of the coarse cell width.
  const double inverse_cell_width_coarse;
  /// Inverse of the fine cell width.
  const double inverse_cell_width_fine;
  /// Global number of coarse cells.
  const int ncells_coarse;
  /// Number of coarse cells per fine cell.
  const int ncells_fine;
  /// Is this mesh running in a mode where it exposes one NP cell per MPI rank.
  bool single_cell_mode;

  /**
   * Construct a mesh over a given MPI communicator with a specified shape.
   *
   * @param comm MPI Communicator to use for decomposition.
   * @param ndim Number of dimensions.
   * @param dims Number of coarse cells in each dimension.
   * @param extent Width of each coarse cell in each dimension.
   * @param subdivision_order Number of times to subdivide each coarse cell to
   * produce the fine cells.
   * @param stencil_width Width of the stencil, in number of cells, used to
   * determine MPI neighbours.
   */
  CartesianHMesh(MPI_Comm comm, const int ndim, std::vector<int> &dims,
                 const double extent = 1.0, const int subdivision_order = 1,
                 const int stencil_width = 0);

  MPI_Comm get_comm();
  int get_ndim();
  std::vector<int> &get_dims();
  int get_subdivision_order();
  double get_cell_width_coarse();
  double get_cell_width_fine();
  double get_inverse_cell_width_coarse();
  double get_inverse_cell_width_fine();
  int get_ncells_coarse();
  int get_ncells_fine();
  std::vector<int> &get_local_communication_neighbours();
  void get_point_in_subdomain(double *point);
  std::shared_ptr<MeshHierarchy> get_mesh_hierarchy();

  /**
   * @returns The number of "cell" NESO-Particles should consider.
   */
  int get_cell_count();

  /**
   * @returns The number of Cartesian cells owned by this MPI rank.
   */
  int get_cart_cell_count();

  /**
   * Convert a mesh index (index_x, index_y, ...) for this cartesian mesh to
   * the format for a MeshHierarchy: (coarse_x, coarse_y,.., fine_x,
   * fine_y,...).
   *
   * @param index_mesh Input tuple index on mesh.
   * @param index_mh Output tuple index on MeshHierarchy.
   */
  void mesh_tuple_to_mh_tuple(const INT *index_mesh, INT *index_mh);

  /**
   *  Free the mesh and any associated communicators.
   */
  void free();

  /**
   * Get a vector of the cells owned by this MPI rank.
   *
   * @returns vector of owned cells in order of the cell ids of the cells.
   */
  std::vector<std::array<int, 3>> get_owned_cells();

  /**
   * Get the global index in tuple form from a local linear cell id.
   *
   * @param linear_cell_index Local linear cell index.
   * @returns Global tuple of cell index.
   */
  std::array<int, 3> get_global_cell_tuple_index(const INT linear_cell_index);

  /**
   * @param face_id Face id to compute owning rank for.
   * @returns Owning rank for passed face id.
   */
  int get_face_id_owning_rank(const INT face_id);

  /**
   * @param index_mesh Cartesian mesh index to find owning rank for.
   * @returns Owning rank of passed mesh index.
   */
  int get_mesh_tuple_owning_rank(const INT *index_mesh);

  /**
   * Convert a face cell id into a face index and local coordinate index for the
   * face.
   *
   * @param[in] face_id Linear face id to convert.
   * @param[in, out] face_index_tuple Index of the face on which the cell lies.
   */
  void get_face_id_as_tuple(const INT face_id, INT *face_index_tuple);

  /**
   * Convert the face geom tuple id to a linear index.
   *
   * @param face_index_tuple Tuple describing the face geom.
   */
  INT get_face_linear_index_from_tuple(const INT *face_index_tuple);

  /**
   * Get the mesh cell as a mesh tuple which has the passed face geometry index
   * as a face.
   *
   * @param[in] face_tuple Tuple which desribes the cell on the face.
   * @param[in, out] Mesh tuple that describes the cell which owns the face.
   */
  void get_mesh_tuple_owning_face_tuple(const INT *face_index_tuple,
                                        INT *mesh_tuple);

  /**
   * Get the coordinates of the vertices of a cell in a form that can be passed
   * directly to NESO::Particles::VTK::UnstructuredCell.
   *
   * @param index Local index of cell to collect vertex coordinates for.
   */
  std::vector<double> get_vtk_cell_points(const INT index);

  /**
   * Get the coordinates of the vertices of a cell in a form that can be passed
   * directly to NESO::Particles::VTK::UnstructuredCell.
   *
   * @param index_tuple Global index in tuple form of cell to collect vertex
   * coordinates for.
   */
  std::vector<double> get_vtk_cell_points(const INT *index_tuple);

  /**
   * Get VTK data for all cells.
   *
   * @returns Vector of VTK data which can be passed to our VTKHDF
   * implementation.
   */
  std::vector<VTK::UnstructuredCell> get_vtk_cell_data();

  /**
   * Get the VTK data for a single face cell.
   *
   * @param face_index_tuple Index of the face cell in tuple form.
   * @returns Vertex coordinates in form VTK::UnstructuredCell can use.
   */
  std::vector<double> get_vtk_face_cell_points(const INT *face_index_tuple);

  /**
   * Get the VTK data for a single face cell.
   *
   * @param face_index Index of the face cell in linear form.
   * @returns Vertex coordinates in form VTK::UnstructuredCell can use.
   */
  std::vector<double> get_vtk_face_cell_points(const INT face_index);

  /**
   * @returns The linear face cell ids of the face cells this MPI rank owns.
   *
   */
  const std::vector<INT> &get_owned_face_cells();

  /**
   * Get VTK data for all face cells.
   *
   * @returns Vector of VTK data which can be passed to our VTKHDF
   * implementation.
   */
  std::vector<VTK::UnstructuredCell> get_vtk_face_cell_data();

  /**
   * @returns A MPI communicator containing ranks that own face cells. This
   * communicator is MPI_COMM_NULL on ranks that do not own face cells.
   */
  MPI_Comm get_face_owning_ranks_comm();
};

typedef std::shared_ptr<CartesianHMesh> CartesianHMeshSharedPtr;

} // namespace NESO::Particles

#endif
