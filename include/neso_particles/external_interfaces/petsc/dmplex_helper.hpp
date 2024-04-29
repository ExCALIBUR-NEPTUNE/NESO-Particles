#ifndef _NESO_PARTICLES_DMPLEX_HELPER_HPP_
#define _NESO_PARTICLES_DMPLEX_HELPER_HPP_

#include "../common/bounding_box.hpp"
#include "dmplex_cell_serialise.hpp"
#include "petsc_common.hpp"
#include <limits>
#include <memory>
#include <set>
#include <vector>

namespace NESO::Particles::PetscInterface {

/**
 * If there are more than 1 MPI ranks distribute the mesh.
 *
 * @param[in, out] dm DMPlex to distribute, original DMPlex is destroyed.
 * @param[in] comm MPI communicator, default MPI_COMM_WORLD.
 */
inline void generic_distribute(DM *dm, MPI_Comm comm = MPI_COMM_WORLD) {
  int size;
  MPICHK(MPI_Comm_size(comm, &size));
  if (size > 1) {
    DM dm_out;
    PETSCCHK(DMPlexDistribute(*dm, 0, nullptr, &dm_out));
    NESOASSERT(dm_out, "Could not distribute mesh.");
    PETSCCHK(DMDestroy(dm));
    *dm = dm_out;
  }
}

/**
 * Setup the coordinate section for a DMPlex. See
 * DMPlexBuildCoordinatesFromCellList.
 *
 * @param dm DMPlex to setup coordinate section for.
 * @param vertex_start Coordinate index for first coordinate.
 * @param vertex_end Coordinate index +1 for last coordinate.
 */
inline void setup_coordinate_section(DM &dm, const PetscInt vertex_start,
                                     const PetscInt vertex_end) {
  PetscInt ndim;
  PETSCCHK(DMGetCoordinateDim(dm, &ndim));
  PetscSection coord_section;
  PETSCCHK(DMGetCoordinateSection(dm, &coord_section));
  PETSCCHK(PetscSectionSetNumFields(coord_section, 1));
  PETSCCHK(PetscSectionSetFieldComponents(coord_section, 0, ndim));
  PETSCCHK(PetscSectionSetChart(coord_section, vertex_start, vertex_end));
  for (PetscInt v = vertex_start; v < vertex_end; ++v) {
    PETSCCHK(PetscSectionSetDof(coord_section, v, ndim));
    PETSCCHK(PetscSectionSetFieldDof(coord_section, v, 0, ndim));
  }
  PETSCCHK(PetscSectionSetUp(coord_section));
}

/**
 * Setup a PETSc vector in which coordinates can be get/set for local mesh. See
 * DMPlexBuildCoordinatesFromCellList.
 *
 * @param[in] dm DMPlex to access coordinates for.
 * @param[in, out] coordinates Vector to setup to use with coordinates.
 * VecDestroy should be called on this vector.
 */
inline void setup_local_coordinate_vector(DM &dm, Vec &coordinates) {
  PetscSection coord_section;
  PETSCCHK(DMGetCoordinateSection(dm, &coord_section));

  // create the actual coordinates vector
  PetscInt coord_size;
  PetscInt ndim;
  PETSCCHK(DMGetCoordinateDim(dm, &ndim));
  PETSCCHK(PetscSectionGetStorageSize(coord_section, &coord_size));
  PETSCCHK(VecCreate(PETSC_COMM_SELF, &coordinates));
  PETSCCHK(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
  PETSCCHK(VecSetSizes(coordinates, coord_size, PETSC_DETERMINE));
  PETSCCHK(VecSetBlockSize(coordinates, ndim));
  PETSCCHK(VecSetType(coordinates, VECSTANDARD));
}

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
   * TODO
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
    return this->map_global_to_local.at(point);
  }

  /**
   * TODO
   */
  HaloDMIndexMapper(std::vector<CellSTDRepresentation> &cells) {
    this->chart_start = 0;
    this->chart_end = 0;

    if (cells.size() > 0) {
      std::map<PetscInt, std::set<PetscInt>> map_depth_to_points;
      for (auto &cx : cells) {
        for (auto &px : cx.point_specs) {
          const auto point = px.first;
          const auto depth = cx.get_point_depth(point);
          map_depth_to_points[depth].insert(point);
        }
      }
      this->depth_max = std::numeric_limits<PetscInt>::min();
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
 * TODO
 */
inline PetscInt dm_from_serialised_cells(
    std::list<DMPlexCellSerialise> &serialised_cells, DM &dm_prototype, DM &dm,
    std::map<PetscInt, std::tuple<int, PetscInt>> &map_local_lid_remote_lid) {

  const PetscInt num_cells = serialised_cells.size();
  std::vector<CellSTDRepresentation> std_rep_cells(num_cells);
  int index = 0;
  for (auto &sc : serialised_cells) {
    std_rep_cells.at(index).deserialise(sc.cell_representation);
  }

  HaloDMIndexMapper index_mapper(std_rep_cells);
  // create the map from the new local dm_halo indices to the original local
  // indices on the remote ranks

  for (auto &sc : serialised_cells) {
    const auto global_index = sc.cell_global_id;
    const PetscInt remote_local_id = sc.cell_local_id;
    const int remote_rank = sc.owning_rank;
    const auto local_local_id =
        index_mapper.get_local_point_index(global_index);
    map_local_lid_remote_lid[local_local_id] = {remote_rank, remote_local_id};
  }

  // Create the new DMPlex.
  PETSCCHK(DMCreate(MPI_COMM_WORLD, &dm));
  PETSCCHK(DMSetType(dm, DMPLEX));

  if (num_cells > 0) {

    PetscInt tmp_int;
    PETSCCHK(DMGetDimension(dm_prototype, &tmp_int));
    PETSCCHK(DMSetDimension(dm, tmp_int));
    PETSCCHK(DMGetCoordinateDim(dm_prototype, &tmp_int));
    const PetscInt ndim_coord = tmp_int;
    PETSCCHK(DMSetCoordinateDim(dm, tmp_int));

    PETSCCHK(
        DMPlexSetChart(dm, index_mapper.chart_start, index_mapper.chart_end));

    for (auto &std_cell : std_rep_cells) {
      for (auto &point_spec : std_cell.point_specs) {
        const PetscInt global_point = point_spec.first;
        const PetscInt local_point =
            index_mapper.get_local_point_index(global_point);
        const PetscInt cone_size = point_spec.second.size();
        PETSCCHK(DMPlexSetConeSize(dm, local_point, cone_size));
      }
    }

    PETSCCHK(DMSetUp(dm));
    std::vector<PetscInt> cone_local;
    for (auto &std_cell : std_rep_cells) {
      for (auto &point_spec : std_cell.point_specs) {
        const PetscInt global_point = point_spec.first;
        const PetscInt local_point =
            index_mapper.get_local_point_index(global_point);
        auto &cone_global = point_spec.second;
        cone_local.clear();
        cone_local.reserve(cone_global.size());
        for (auto gx : cone_global) {
          cone_local.push_back(index_mapper.get_local_point_index(gx));
        }
        PETSCCHK(DMPlexSetCone(dm, local_point, cone_local.data()));
      }
    }

    PETSCCHK(DMPlexSymmetrize(dm));
    PETSCCHK(DMPlexStratify(dm));

    PetscInt vertex_start, vertex_end;
    index_mapper.get_depth_stratum(0, &vertex_start, &vertex_end);
    setup_coordinate_section(dm, vertex_start, vertex_end);
    Vec coordinates;
    PetscScalar *coords;
    PetscInterface::setup_local_coordinate_vector(dm, coordinates);
    PETSCCHK(VecGetArray(coordinates, &coords));

    // write coordinates of vertices
    for (auto &std_cell : std_rep_cells) {
      for (const auto &point_vertex : std_cell.vertices) {
        const PetscInt global_point = point_vertex.first;
        const PetscInt local_point =
            index_mapper.get_local_point_index(global_point);
        const PetscInt vertex_index = local_point - vertex_start;
        for (PetscInt dimx = 0; dimx < ndim_coord; dimx++) {
          const PetscScalar value = point_vertex.second.at(dimx);
          coords[vertex_index * ndim_coord + dimx] = value;
        }
      }
    }

    PETSCCHK(VecRestoreArray(coordinates, &coords));
    PETSCCHK(DMSetCoordinatesLocal(dm, coordinates));
    PETSCCHK(VecDestroy(&coordinates));

    DM dm_interpolated;
    PETSCCHK(DMPlexInterpolate(dm, &dm_interpolated));
    PETSCCHK(DMDestroy(&dm));
    dm = dm_interpolated;

    // Create the maps from the created dm cells to the original cell ids and
    // owning ranks.
  }

  return num_cells;
}

/**
 * TODO
 */
class DMPlexHelper {
protected:
  inline void check_valid_cell(const PetscInt cell) const {
    NESOASSERT((cell > -1) && (cell < (this->cell_end - this->cell_start)),
               "Bad cell index passed.");
  }

public:
  MPI_Comm comm;
  DM dm;
  PetscInt ndim;
  IS global_cell_numbers;
  IS global_vertex_numbers;
  IS global_point_numbers;
  PetscInt cell_start;
  PetscInt cell_end;

  /**
   * TODO
   */
  inline DMPlexCellSerialise get_copyable_cell(const PetscInt cell) {
    DMPlexCellSerialise cs;

    int rank;
    MPICHK(MPI_Comm_rank(this->comm, &rank));
    DMPolytopeType cell_type;
    PETSCCHK(DMPlexGetCellType(dm, cell, &cell_type));

    auto lambda_rename = [&](PetscInt cell) -> PetscInt {
      return this->get_point_global_index(cell);
    };
    std::function<PetscInt(PetscInt)> rename_function = lambda_rename;
    auto spec =
        PetscInterface::get_cell_specification(this->dm, cell, rename_function);

    std::vector<std::byte> cell_representation;
    spec.serialise(cell_representation);

    cs.cell_local_id = cell;
    cs.cell_global_id = get_point_global_index(cell);
    cs.owning_rank = rank;
    cs.cell_type = cell_type;
    cs.cell_representation = cell_representation;

    return cs;
  }

  /**
   * TODO
   */
  DMPlexHelper(MPI_Comm comm, DM dm) : comm(comm), dm(dm) {
    DMPlexInterpolatedFlag interpolated;
    PETSCCHK(DMPlexIsInterpolated(this->dm, &interpolated));
    NESOASSERT(interpolated == DMPLEX_INTERPOLATED_FULL,
               "Expected fully interpolated mesh.");
    PETSCCHK(DMGetCoordinateDim(this->dm, &this->ndim));
    PETSCCHK(DMPlexGetHeightStratum(this->dm, 0, &this->cell_start,
                                    &this->cell_end));
    PETSCCHK(DMPlexGetCellNumbering(this->dm, &this->global_cell_numbers));
    PETSCCHK(DMPlexGetVertexNumbering(this->dm, &this->global_vertex_numbers));
    PETSCCHK(DMPlexCreatePointNumbering(this->dm, &this->global_point_numbers));
  }

  /**
   * Remove the negation from point indices which DMPlex uses to denote global
   * vs local indices.
   *
   * @param c Input index.
   * @returns c if c > -1 else ((c * (-1)) - 1)
   */
  inline PetscInt signed_global_id_to_global_id(const PetscInt c) {
    return (c > -1) ? c : ((c * (-1)) - 1);
  }

  /**
   * Get the global index of a point from the local point index.
   *
   * @param point Local point index.
   * @returns Global point index.
   */
  inline PetscInt get_point_global_index(const PetscInt point) {
    PetscInt global_point;
    const PetscInt *ptr;
    PETSCCHK(ISGetIndices(this->global_point_numbers, &ptr));
    global_point = ptr[point];
    PETSCCHK(ISRestoreIndices(this->global_point_numbers, &ptr));
    return signed_global_id_to_global_id(global_point);
  }

  /**
   * Get the global index of a cell from the local cell index.
   *
   * @param cell Local cell index.
   * @returns Global cell index.
   */
  inline PetscInt get_cell_global_index(const PetscInt cell) {
    this->check_valid_cell(cell);

    PetscInt global_cell;
    const PetscInt *ptr;
    PETSCCHK(ISGetIndices(this->global_cell_numbers, &ptr));
    global_cell = ptr[cell];
    PETSCCHK(ISRestoreIndices(this->global_cell_numbers, &ptr));
    NESOASSERT(global_cell > -1,
               "Cell index was negative indicating a remote cell.");

    return global_cell;
  }

  /**
   * Get the global index of a vertex from the local vertex index.
   *
   * @param vertex Local vertex index.
   * @returns Global vertex index.
   */
  inline PetscInt get_vertex_global_index(const PetscInt vertex) {

    PetscInt global_vertex;
    const PetscInt *ptr;
    PETSCCHK(ISGetIndices(this->global_vertex_numbers, &ptr));
    global_vertex = ptr[vertex];
    PETSCCHK(ISRestoreIndices(this->global_vertex_numbers, &ptr));
    NESOASSERT(global_vertex > -1,
               "Vertex index was negative indicating a remote vertex.");

    return global_vertex;
  }

  /**
   * Get a bounding box for a mesh cell (assumes linear mesh).
   *
   * @param cell Local cell index.
   * @returns Bounding box for cell.
   */
  inline ExternalCommon::BoundingBoxSharedPtr
  get_cell_bounding_box(const PetscInt cell) {
    this->check_valid_cell(cell);

    std::vector<REAL> bb = {
        std::numeric_limits<REAL>::max(), std::numeric_limits<REAL>::max(),
        std::numeric_limits<REAL>::max(), std::numeric_limits<REAL>::min(),
        std::numeric_limits<REAL>::min(), std::numeric_limits<REAL>::min()};

    for (int dx = this->ndim; dx < 3; dx++) {
      bb[dx] = 0.0;
      bb[dx + 3] = 0.0;
    }

    const PetscScalar *array;
    PetscScalar *coords = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;
    PETSCCHK(DMPlexGetCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                      &num_coords, &array, &coords));
    NESOASSERT(coords != nullptr, "No vertices returned for cell.");
    const PetscInt num_verts = num_coords / ndim;
    for (PetscInt vx = 0; vx < num_verts; vx++) {
      for (PetscInt dimx = 0; dimx < this->ndim; dimx++) {
        const REAL cx = coords[vx * ndim + dimx];
        bb[dimx] = std::min(bb[dimx], cx);
        bb[dimx + 3] = std::max(bb[dimx + 3], cx);
      }
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                          &num_coords, &array, &coords));
    return std::make_shared<ExternalCommon::BoundingBox>(bb);
  }

  /**
   * Get average of the vertices of a cell.
   *
   * @param[in] cell Local cell index.
   * @param[in, out] average Vector of average of vertices.
   */
  inline void get_cell_vertex_average(const PetscInt cell,
                                      std::vector<REAL> &average) {
    this->check_valid_cell(cell);
    NESOASSERT(average.size() == this->ndim,
               "Missmatch between vector size and number of dimensions");

    std::fill(average.begin(), average.end(), 0.0);
    const PetscScalar *array;
    PetscScalar *coords = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;
    PETSCCHK(DMPlexGetCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                      &num_coords, &array, &coords));
    NESOASSERT(coords != nullptr, "No vertices returned for cell.");
    const PetscInt num_verts = num_coords / ndim;
    for (PetscInt vx = 0; vx < num_verts; vx++) {
      for (PetscInt dimx = 0; dimx < this->ndim; dimx++) {
        const REAL cx = coords[vx * ndim + dimx];
        average.at(dimx) += cx;
      }
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                          &num_coords, &array, &coords));

    const REAL tmp_factor = 1.0 / ((REAL)num_verts);
    for (int dx = 0; dx < this->ndim; dx++) {
      average.at(dx) *= tmp_factor;
    }
  }

  /**
   * Determine if mesh contains a point.
   *
   * @param[in] point Point to test.
   * @returns Negative value if point not located, otherwise owning cell.
   */
  inline int contains_point(std::vector<PetscScalar> &point) {
    const PetscInt ndim = this->ndim;
    NESOASSERT(point.size() == ndim,
               "Miss-match in point size and mesh dimension.");
    Vec v;
    PETSCCHK(VecCreate(MPI_COMM_SELF, &v));
    PETSCCHK(VecSetSizes(v, ndim, ndim));
    PETSCCHK(VecSetBlockSize(v, ndim));
    PETSCCHK(VecSetFromOptions(v));
    PetscScalar *v_ptr;
    PETSCCHK(VecGetArrayWrite(v, &v_ptr));
    for (int dimx = 0; dimx < ndim; dimx++) {
      v_ptr[dimx] = point.at(dimx);
    }
    PETSCCHK(VecRestoreArrayWrite(v, &v_ptr));
    PetscSF cell_sf = nullptr;
    PETSCCHK(DMLocatePoints(dm, v, DM_POINTLOCATION_NONE, &cell_sf));
    const PetscSFNode *cells;
    PetscInt n_found;
    const PetscInt *found;
    PETSCCHK(PetscSFGetGraph(cell_sf, NULL, &n_found, &found, &cells));
    return cells[0].index;
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
