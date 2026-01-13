#ifndef _NESO_PARTICLES_DMPLEX_HELPER_HPP_
#define _NESO_PARTICLES_DMPLEX_HELPER_HPP_

#include "../../containers/cell_dat_const.hpp"
#include "../common/bounding_box.hpp"
#include "../vtk/vtk.hpp"
#include "dmplex_cell_serialise.hpp"
#include "petsc_common.hpp"
#include <limits>
#include <memory>
#include <set>
#include <vector>

namespace NESO::Particles::PetscInterface {

constexpr static char face_sets_label[] = "Face Sets";

/**
 * If there are more than 1 MPI ranks distribute the mesh.
 *
 * @param[in, out] dm DMPlex to distribute, original DMPlex is destroyed.
 * @param[in] comm MPI communicator, default MPI_COMM_WORLD.
 * @param[in] overlap Optional overlap to pass to PETSc (default 0).
 */
void generic_distribute(DM *dm, MPI_Comm comm = MPI_COMM_WORLD,
                        const PetscInt overlap = 0, PetscSF *sf = nullptr);
/**
 * Setup the coordinate section for a DMPlex. See
 * DMPlexBuildCoordinatesFromCellList.
 *
 * @param dm DMPlex to setup coordinate section for.
 * @param vertex_start Coordinate index for first coordinate.
 * @param vertex_end Coordinate index +1 for last coordinate.
 */
void setup_coordinate_section(DM &dm, const PetscInt vertex_start,
                              const PetscInt vertex_end);

/**
 * Setup a PETSc vector in which coordinates can be get/set for local mesh. See
 * DMPlexBuildCoordinatesFromCellList.
 *
 * @param[in] dm DMPlex to access coordinates for.
 * @param[in, out] coordinates Vector to setup to use with coordinates.
 * VecDestroy should be called on this vector.
 */
void setup_local_coordinate_vector(DM &dm, Vec &coordinates);

/**
 * Class to determine new local indices from global indices when constructing
 * halos.
 */
struct HaloDMIndexMapper {
  PetscInt depth_min;
  PetscInt depth_max;
  PetscInt chart_start;
  PetscInt chart_end;
  std::vector<PetscInt> depth_starts;
  std::vector<PetscInt> depth_ends;
  std::map<PetscInt, PetscInt> map_global_to_local;

  /**
   * Get the start and end points for a given depth in the DMPlex.
   *
   * @param[in] depth Depth to get stratum for.
   * @param[in, out] start First PETSc point index for given depth.
   * @param[in, out] end Last PETSc point index plus one for given depth.
   */
  inline void get_depth_stratum(const PetscInt depth, PetscInt *start,
                                PetscInt *end) {
    *start = this->depth_starts.at(depth);
    *end = this->depth_ends.at(depth);
  }

  /**
   * Get the new local point index for a global point index.
   *
   * @param point Global point index.
   * @returns Local point index for a new DM.
   */
  inline PetscInt get_local_point_index(const PetscInt point) {
    const auto local_point = this->map_global_to_local.at(point);
    return local_point;
  }

  /**
   * Create an instance from STD representation on cells, e.g. after cells have
   * been communicated to ranks for the purpose of building halos.
   *
   * @param cells Vector of cells which require new local indices for all the
   * points contained.
   */
  HaloDMIndexMapper(std::vector<CellSTDRepresentation> &cells) {
    this->chart_start = 0;
    this->chart_end = 0;

    if (cells.size() > 0) {

      std::map<PetscInt, std::set<PetscInt>> map_depth_to_points;
      for (auto &cx : cells) {
        for (auto &px : cx.point_cones) {
          const auto point = px.first;
          const auto depth = cx.get_point_depth(point);
          map_depth_to_points[depth].insert(point);
        }
      }
      this->depth_max = std::numeric_limits<PetscInt>::lowest();
      this->depth_min = std::numeric_limits<PetscInt>::max();
      for (auto &depth_points : map_depth_to_points) {
        this->depth_max = std::max(this->depth_max, depth_points.first);
        this->depth_min = std::min(this->depth_min, depth_points.first);
      }
      NESOASSERT(this->depth_min == 0,
                 "Expected minium depth to be 0 for vertices.");

      // Get the ranges for the local indices for the new DM
      std::vector<PetscInt> starting_indices(this->depth_max + 1);
      this->depth_starts.resize(this->depth_max + 1);
      this->depth_ends.resize(this->depth_max + 1);
      this->depth_starts.at(0) = 0;
      this->depth_ends.at(0) = map_depth_to_points.at(0).size();
      starting_indices.at(0) = 0;
      for (int depth = 1; depth <= this->depth_max; depth++) {
        const PetscInt prev_end = this->depth_ends.at(depth - 1);
        this->depth_starts.at(depth) = prev_end;
        this->depth_ends.at(depth) =
            prev_end + map_depth_to_points.at(depth).size();
        starting_indices.at(depth) = prev_end;
      }

      // Get the new indices for points
      for (auto &depth_points : map_depth_to_points) {
        const PetscInt depth = depth_points.first;
        for (const PetscInt global_point : depth_points.second) {
          const PetscInt local_point = starting_indices.at(depth)++;
          this->map_global_to_local[global_point] = local_point;
          this->chart_end++;
        }
      }

      for (int depth = 0; depth <= this->depth_max; depth++) {
        const PetscInt end_index = this->depth_ends.at(depth);
        NESOASSERT(end_index == starting_indices.at(depth),
                   "Error mapping old indices to new indices");
      }
      NESOASSERT(this->chart_end ==
                     this->depth_ends.at(this->depth_ends.size() - 1),
                 "Error mapping chart start/end=.");
    }
  }
};

/**
 * Create a new DMPlex from the serialised cells. These serialised cells are
 * originally from the prototype DMPlex and have been serialised and
 * communicated between MPI ranks.
 *
 * @param[in] serialised_cells Serialised cells to create a new DMPlex from.
 * @param[in] dm_prototype The DMPlex from which the serialised cells are from.
 * @param[in, out] dm The new DMPlex to create from the serialised cells.
 * @param[in, out] map_local_lid_remote_lid A map from the new local cell
 * indices to a tuple of {original owning rank, original local id on owning
 * rank, global petsc index of cell}.
 * @returns True if the constructed DMPlex is not empty otherwise false.
 */
bool dm_from_serialised_cells(
    std::list<DMPlexCellSerialise> &serialised_cells, DM &dm_prototype, DM &dm,
    std::map<PetscInt, std::tuple<int, PetscInt, PetscInt>>
        &map_local_lid_remote_lid);

/**
 * Helper class that wraps a PETSc DMPlex and simplifies common operations.
 */
class DMPlexHelper {
protected:
  IS global_point_numbers;
  PetscInt point_start;
  PetscInt point_end;
  PetscInt cell_start;
  PetscInt cell_end;
  std::vector<PetscInt> map_np_to_petsc;
  std::map<PetscInt, PetscInt> map_petsc_to_np;
  std::map<PetscInt, PetscInt> map_gobal_point_to_local_point;
  double volume;
  int ncells_global{-1};

  inline void check_valid_local_cell(const PetscInt cell) const {
    NESOASSERT((cell > -1) && (cell < this->ncells),
               "Bad NESO-Particles cell index passed: Out of range.");
  }
  inline void check_valid_petsc_cell(const PetscInt cell) const {
    NESOASSERT((this->cell_start <= cell) && (cell < this->cell_end),
               "Bad DMPlex cell index passed: Not in range.");
    NESOASSERT(this->map_petsc_to_np.count(cell),
               "Bad DMPlex cell index passed: PETSc cell index is not local.");
  }
  inline void check_valid_petsc_point(const PetscInt petsc_index) const {
    NESOASSERT((this->point_start <= petsc_index) &&
                   (petsc_index < this->point_end),
               "Bad point index passed: " + std::to_string(petsc_index) +
                   ". Not in range.");
  }

  inline PetscInt internal_get_point_global_index(const PetscInt point) {
    PetscInt global_point;
    const PetscInt *ptr;
    PETSCCHK(ISGetIndices(this->global_point_numbers, &ptr));
    global_point = ptr[point - this->point_start];
    PETSCCHK(ISRestoreIndices(this->global_point_numbers, &ptr));
    return global_point;
  }

  ExternalCommon::BoundingBoxSharedPtr bounding_box;

public:
  MPI_Comm comm;
  DM dm;
  PetscInt ndim;
  PetscInt ncells;

  /**
   * Get a serialisable representation of a cell in the DMPlex.
   *
   * @param local_index Local index of the cell to get a representation of.
   * @returns Serialisable representation of the cell.
   */
  DMPlexCellSerialise get_copyable_cell(const PetscInt local_index);

  /**
   * Free the helper. Must be called collectively on the communicator.
   */
  void free();

  /**
   * Construct helper class from DMPlex. Collective on the communicator.
   *
   * @param comm MPI communicator the DMPlex is constructed on.
   * @param dm Input DMPlex to wrap.
   */
  DMPlexHelper(MPI_Comm comm, DM dm);

  /**
   * @returns the number of cells.
   */
  int get_cell_count();

  /**
   * @returns the total number of cells. Collective on the communicator.
   */
  int get_global_cell_count();

  /**
   * Convert a local cell id into a DMPlex cell id.
   *
   * @param local_index Local index in [0, num_cells).
   * @returns DMPlex point index in [cell_start, cell_end).
   */
  PetscInt get_dmplex_cell_index(const PetscInt local_index);

  /**
   * Convert a DMPlex cell id into a local cell id.
   *
   * @param petsc_index Local index in [cell_start, cell_end).
   * @returns Local point index in [0, num_cells).
   */
  PetscInt get_local_cell_index(const PetscInt petsc_index);

  /**
   * Remove the negation from point indices which DMPlex uses to denote global
   * vs local indices.
   *
   * @param c Input index.
   * @returns c if c > -1 else ((c * (-1)) - 1)
   */
  PetscInt signed_global_id_to_global_id(const PetscInt c);

  /**
   * Get the global index of a point from the local point index.
   *
   * @param point Local point index.
   * @param signed_point Optionally return the original signed global point
   * index.
   * @returns Global point index.
   */
  PetscInt get_point_global_index(const PetscInt point,
                                  const bool signed_point = false);

  /**
   * Get the local point index from a global point index if the point is owned
   * by this rank.
   *
   * @param global_point_index Global point index to retrieve local point index
   * for.
   * @returns Local point index.
   */
  PetscInt get_local_point_from_global_point(const PetscInt global_point_index);

  /**
   * Get a bounding box for the cells on this MPI rank.
   *
   * @returns Bounding box for cells in DMPlex.
   */
  ExternalCommon::BoundingBoxSharedPtr get_bounding_box();

  /**
   * Get a bounding box for a mesh cell (assumes linear mesh).
   *
   * @param cell Local cell index.
   * @returns Bounding box for cell.
   */
  ExternalCommon::BoundingBoxSharedPtr
  get_cell_bounding_box(const PetscInt cell);

  /**
   * Get the vertices of an edge using a PETSc index.
   *
   * @param[in] petsc_index PETSc point index.
   * @param[in, out] vertices Vector of vertices.
   */
  void get_generic_vertices(const PetscInt petsc_index,
                            std::vector<std::vector<REAL>> &vertices);

  /**
   * Get the vertices of a cell.
   *
   * @param[in] cell Local cell index.
   * @param[in, out] vertices Vector of vertices.
   */
  void get_cell_vertices(const PetscInt cell,
                         std::vector<std::vector<REAL>> &vertices);

  /**
   * Get average of the vertices of a cell.
   *
   * @param[in] cell Local cell index.
   * @param[in, out] average Vector of average of vertices.
   */
  void get_cell_vertex_average(const PetscInt cell, std::vector<REAL> &average);

  /**
   * Determine if mesh contains a point.
   *
   * @param[in] point Point to test.
   * @returns Negative value if point not located, otherwise owning cell.
   */
  int contains_point(std::vector<PetscScalar> &point);

  /**
   * Determine if mesh cell contains a point.
   *
   * @param[in] index Local cell index.
   * @param[in] point Point to test.
   * @returns true if point contains cell.
   */
  bool cell_contains_point(const PetscInt index,
                           std::vector<PetscScalar> &point);

  /**
   * @returns The number of labels in the DMPlex.
   */
  PetscInt get_num_labels();

  /**
   * @returns The DMLabel for "Face Sets"
   */
  DMLabel get_face_sets_label();

  /**
   * @param index Index of label to get name of.
   * @returns The label name corresponding to an index.
   */
  std::string get_label_name(const PetscInt index);

  /**
   * Get boundary points start and end. In 2D the boundary point types are
   * edges. In 3D the boundary point types are faces. Returns [start, end).
   *
   * @param[in, out] start First boundary point.
   * @param[in, out] end Last boundary point plus one.
   */
  void get_boundary_stratum(PetscInt *start, PetscInt *end);

  /**
   * @returns Map from face sets int label to DMPlex points with that label.
   */
  std::map<PetscInt, std::vector<PetscInt>> get_face_sets();

  /**
   * Write the DM to a file for visualisation in paraview.
   *
   * @param filename Filename for VTK file.
   */
  void write_vtk(const std::string filename);

  /**
   * Get VTK data for all cells.
   *
   * @returns Vector of VTK data which can be passed to our VTKHDF
   * implementation.
   */
  std::vector<VTK::UnstructuredCell> get_vtk_cell_data();

  /**
   * Print to stdout information about the held DMPlex.
   */
  void print();

  /**
   * Get the volume of a cell in the local mesh.
   *
   * @param cell Local index of cell.
   * @returns Volume of cell.
   */
  REAL get_cell_volume(const int index);

  /**
   * @returns The total volume of the mesh. Must be called collectively on the
   * communicator.
   */
  REAL get_volume();
};

/**
 * Get the number of cell vertices and cell vertices in a CellDatConst.
 *
 * @returns CellDatConst for number of cell vertices and cell vertices.
 */
std::tuple<std::shared_ptr<CellDatConst<int>>,
           std::shared_ptr<CellDatConst<REAL>>>
get_cell_vertices_cdc(SYCLTargetSharedPtr sycl_target,
                      std::shared_ptr<DMPlexHelper> dmh);

/**
 * Get the map from global cell index to owning rank. Assumes that the N global
 * cell indices live in [a, a+N) and returns a vector of MPI ranks and a. Must
 * be called collectively on the communicator of the DMPlex.
 *
 * @param dm DMPlex to retrieve owning ranks for.
 * @returns The offset a and the vector that holds the MPI rank that owns cell i
 * + a in element i.
 */
std::pair<int, std::vector<int>>
get_map_from_global_cell_points_to_ranks(DM dm);

} // namespace NESO::Particles::PetscInterface

#endif
