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
 * @param[in] overlap Optional overlap to pass to PETSc (default 0).
 */
inline void generic_distribute(DM *dm, MPI_Comm comm = MPI_COMM_WORLD,
                               const PetscInt overlap = 0) {
  int size;
  MPICHK(MPI_Comm_size(comm, &size));
  if (size > 1) {
    DM dm_out;
    PETSCCHK(DMPlexDistribute(*dm, overlap, nullptr, &dm_out));
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
    const auto local_point = this->map_global_to_local.at(point);
    return local_point;
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
 * TODO
 */
inline bool
dm_from_serialised_cells(std::list<DMPlexCellSerialise> &serialised_cells,
                         DM &dm_prototype, DM &dm,
                         std::map<PetscInt, std::tuple<int, PetscInt, PetscInt>>
                             &map_local_lid_remote_lid) {

  const PetscInt num_cells = serialised_cells.size();
  std::vector<CellSTDRepresentation> std_rep_cells(num_cells);
  int index = 0;
  for (auto &sc : serialised_cells) {
    std_rep_cells.at(index).deserialise(sc.cell_representation);
    index++;
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
    map_local_lid_remote_lid[local_local_id] = {remote_rank, remote_local_id,
                                                global_index};
  }

  // Create the new DMPlex.
  PETSCCHK(DMCreate(PETSC_COMM_SELF, &dm));
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

    std::set<PetscInt> points_set;
    for (auto &std_cell : std_rep_cells) {
      for (auto &point_spec : std_cell.point_specs) {
        const PetscInt global_point = point_spec.first;
        if (!points_set.count(global_point)) {
          points_set.insert(global_point);
          const PetscInt local_point =
              index_mapper.get_local_point_index(global_point);
          const PetscInt cone_size = point_spec.second.size();
          PETSCCHK(DMPlexSetConeSize(dm, local_point, cone_size));
        }
      }
    }

    PETSCCHK(DMSetUp(dm));
    std::vector<PetscInt> cone_local;
    points_set.clear();
    for (auto &std_cell : std_rep_cells) {
      for (auto &point_spec : std_cell.point_specs) {
        const PetscInt global_point = point_spec.first;
        if (!points_set.count(global_point)) {
          points_set.insert(global_point);
          const PetscInt local_point =
              index_mapper.get_local_point_index(global_point);
          auto &cone_global = point_spec.second;
          cone_local.clear();
          cone_local.reserve(cone_global.size());
          for (auto gx : cone_global) {
            cone_local.push_back(index_mapper.get_local_point_index(gx));
          }
          PETSCCHK(DMPlexSetCone(dm, local_point, cone_local.data()));
          if (!map_local_lid_remote_lid.count(local_point)) {
            map_local_lid_remote_lid[local_point] = {-1, -1, global_point};
          }
        }
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
  }

  return num_cells > 0;
}

/**
 * TODO
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
  double volume;

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
   * TODO
   */
  inline DMPlexCellSerialise get_copyable_cell(const PetscInt local_index) {
    this->check_valid_local_cell(local_index);
    const PetscInt point = this->map_np_to_petsc.at(local_index);
    DMPlexCellSerialise cs;

    int rank;
    MPICHK(MPI_Comm_rank(this->comm, &rank));
    DMPolytopeType cell_type;
    PETSCCHK(DMPlexGetCellType(dm, point, &cell_type));

    auto lambda_rename = [&](PetscInt cell) -> PetscInt {
      return this->get_point_global_index(cell);
    };
    std::function<PetscInt(PetscInt)> rename_function = lambda_rename;
    auto spec = PetscInterface::get_cell_specification(this->dm, point,
                                                       rename_function);

    std::vector<std::byte> cell_representation;
    spec.serialise(cell_representation);

    cs.cell_local_id = local_index;
    cs.cell_global_id = lambda_rename(point);
    cs.owning_rank = rank;
    cs.cell_type = cell_type;
    cs.cell_representation = cell_representation;

    return cs;
  }

  /**
   * TODO
   */
  DMPlexHelper(MPI_Comm comm, DM dm)
      : comm(comm), dm(dm), bounding_box(nullptr), volume(-1.0) {
    DMPlexInterpolatedFlag interpolated;
    PETSCCHK(DMPlexIsInterpolated(this->dm, &interpolated));
    NESOASSERT(interpolated == DMPLEX_INTERPOLATED_FULL,
               "Expected fully interpolated mesh.");
    PETSCCHK(DMGetCoordinateDim(this->dm, &this->ndim));
    PETSCCHK(DMPlexGetHeightStratum(this->dm, 0, &this->cell_start,
                                    &this->cell_end));
    PETSCCHK(DMPlexGetChart(this->dm, &this->point_start, &this->point_end));
    PETSCCHK(DMPlexCreatePointNumbering(this->dm, &this->global_point_numbers));

    this->map_np_to_petsc.clear();
    PetscInt ix = 0;
    for (int cx = this->cell_start; cx < this->cell_end; cx++) {
      auto global = this->internal_get_point_global_index(cx);
      if (global > -1) {
        this->map_np_to_petsc.push_back(cx);
        this->map_petsc_to_np[cx] = ix++;
      }
    }

    NESOASSERT(ix == this->map_petsc_to_np.size(), "Size missmatch.");
    NESOASSERT(ix == this->map_np_to_petsc.size(), "Size missmatch.");
    this->ncells = this->map_np_to_petsc.size();
    NESOASSERT(this->ncells, "A rank has zero cells.");
  }

  /**
   * @returns the number of cells.
   */
  inline int get_cell_count() { return this->ncells; }

  /**
   * Convert a local cell id into a DMPlex cell id.
   *
   * @param local_index Local index in [0, num_cells).
   * @returns DMPlex point index in [cell_start, cell_end).
   */
  inline PetscInt get_dmplex_cell_index(const PetscInt local_index) {
    // TODO MAKE PROTECTED
    this->check_valid_local_cell(local_index);
    const auto index = this->map_np_to_petsc.at(local_index);
    this->check_valid_petsc_cell(index);
    return index;
  }

  /**
   * Convert a DMPlex cell id into a local cell id.
   *
   * @param petsc_index Local index in [cell_start, cell_end).
   * @returns Local point index in [0, num_cells).
   */
  inline PetscInt get_local_cell_index(const PetscInt petsc_index) {
    // TODO MAKE PROTECTED
    this->check_valid_petsc_cell(petsc_index);
    const auto index = this->map_petsc_to_np.at(petsc_index);
    this->check_valid_local_cell(index);
    return index;
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
   * @param signed_point Optionally return the original signed global point
   * index.
   * @returns Global point index.
   */
  inline PetscInt get_point_global_index(const PetscInt point,
                                         const bool signed_point = false) {
    NESOASSERT((this->point_start <= point) && (point < this->point_end),
               "Bad point passed.");
    PetscInt global_point = this->internal_get_point_global_index(point);

    if (signed_point) {
      return global_point;
    } else {
      return signed_global_id_to_global_id(global_point);
    }
  }

  /**
   * Get a bounding box for the cells on this MPI rank.
   *
   * @returns Bounding box for cells in DMPlex.
   */
  inline ExternalCommon::BoundingBoxSharedPtr get_bounding_box() {
    // Create the bounding box on first use.
    if (!this->bounding_box) {
      auto bb = std::make_shared<ExternalCommon::BoundingBox>();
      const auto num_cells = this->get_cell_count();
      for (int cellx = 0; cellx < num_cells; cellx++) {
        bb->expand(this->get_cell_bounding_box(cellx));
      }
      this->bounding_box = bb;
    }
    return this->bounding_box;
  }

  /**
   * Get a bounding box for a mesh cell (assumes linear mesh).
   *
   * @param cell Local cell index.
   * @returns Bounding box for cell.
   */
  inline ExternalCommon::BoundingBoxSharedPtr
  get_cell_bounding_box(const PetscInt cell) {
    this->check_valid_local_cell(cell);

    std::vector<REAL> bb = {std::numeric_limits<REAL>::max(),
                            std::numeric_limits<REAL>::max(),
                            std::numeric_limits<REAL>::max(),
                            std::numeric_limits<REAL>::lowest(),
                            std::numeric_limits<REAL>::lowest(),
                            std::numeric_limits<REAL>::lowest()};

    for (int dx = this->ndim; dx < 3; dx++) {
      bb[dx] = 0.0;
      bb[dx + 3] = 0.0;
    }

    const PetscScalar *array;
    PetscScalar *coords = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;
    const PetscInt petsc_index = this->map_np_to_petsc.at(cell);
    this->check_valid_petsc_cell(petsc_index);
    PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                      &array, &coords));
    NESOASSERT(coords != nullptr, "No vertices returned for cell.");
    const PetscInt num_verts = num_coords / ndim;
    for (PetscInt vx = 0; vx < num_verts; vx++) {
      for (PetscInt dimx = 0; dimx < this->ndim; dimx++) {
        const REAL cx = coords[vx * ndim + dimx];
        bb[dimx] = std::min(bb[dimx], cx);
        bb[dimx + 3] = std::max(bb[dimx + 3], cx);
      }
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                          &array, &coords));
    return std::make_shared<ExternalCommon::BoundingBox>(bb);
  }

  /**
   * Get the vertices of a cell.
   *
   * @param[in] cell Local cell index.
   * @param[in, out] vertices Vector of vertices.
   */
  inline void get_cell_vertices(const PetscInt cell,
                                std::vector<std::vector<REAL>> &vertices) {
    this->check_valid_local_cell(cell);

    const PetscScalar *array;
    PetscScalar *coords = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;
    const PetscInt petsc_index = this->map_np_to_petsc.at(cell);
    this->check_valid_petsc_cell(petsc_index);
    PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                      &array, &coords));
    NESOASSERT(coords != nullptr, "No vertices returned for cell.");
    const PetscInt num_verts = num_coords / ndim;

    vertices.clear();
    vertices.reserve(num_verts);
    for (PetscInt vx = 0; vx < num_verts; vx++) {
      std::vector<REAL> tmp(ndim);
      for (PetscInt dimx = 0; dimx < this->ndim; dimx++) {
        const REAL cx = coords[vx * ndim + dimx];
        tmp.at(dimx) = cx;
      }
      vertices.push_back(tmp);
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                          &array, &coords));
  }

  /**
   * Get average of the vertices of a cell.
   *
   * @param[in] cell Local cell index.
   * @param[in, out] average Vector of average of vertices.
   */
  inline void get_cell_vertex_average(const PetscInt cell,
                                      std::vector<REAL> &average) {
    this->check_valid_local_cell(cell);
    NESOASSERT(average.size() == this->ndim,
               "Missmatch between vector size and number of dimensions");

    std::fill(average.begin(), average.end(), 0.0);
    std::vector<std::vector<REAL>> vertices;
    this->get_cell_vertices(cell, vertices);
    const int num_verts = vertices.size();
    for (PetscInt vx = 0; vx < num_verts; vx++) {
      for (PetscInt dimx = 0; dimx < this->ndim; dimx++) {
        const REAL cx = vertices.at(vx).at(dimx);
        average.at(dimx) += cx;
      }
    }
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

  /**
   * Determine if mesh cell contains a point.
   *
   * @param[in] index Local cell index.
   * @param[in] point Point to test.
   * @returns true if point contains cell.
   */
  inline bool cell_contains_point(const PetscInt index,
                                  std::vector<PetscScalar> &point) {
    const PetscInt ndim = this->ndim;
    NESOASSERT(point.size() == ndim,
               "Miss-match in point size and mesh dimension.");
    NESOASSERT(2 == ndim, "Only implemented in 2D.");
    this->check_valid_local_cell(index);
    const PetscInt petsc_index = this->map_np_to_petsc.at(index);
    bool contained = false;
    const PetscScalar x0 = point.at(0);
    const PetscScalar x1 = point.at(1);

    const PetscScalar *tmp;
    PetscScalar *vertices = nullptr;
    PetscReal x = PetscRealPart(point[0]);
    PetscReal y = PetscRealPart(point[1]);
    PetscInt num_crossings = 0, num_coords;
    PetscBool is_dg;

    PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                      &tmp, &vertices));
    const int num_faces = num_coords / 2;
    int faces[8];
    if (num_coords == 8) {
      // {0, 1, 1, 2, 2, 3, 3, 0};
      faces[0] = 0;
      faces[1] = 1;
      faces[2] = 1;
      faces[3] = 2;
      faces[4] = 2;
      faces[5] = 3;
      faces[6] = 3;
      faces[7] = 0;
    } else {
      // {0, 1, 1, 2, 2, 0};
      faces[0] = 0;
      faces[1] = 1;
      faces[2] = 1;
      faces[3] = 2;
      faces[4] = 2;
      faces[5] = 0;
    }

    for (int facex = 0; facex < num_faces; facex++) {
      REAL xi = vertices[faces[2 * facex + 0] * 2 + 0];
      REAL yi = vertices[faces[2 * facex + 0] * 2 + 1];
      REAL xj = vertices[faces[2 * facex + 1] * 2 + 0];
      REAL yj = vertices[faces[2 * facex + 1] * 2 + 1];
      // Is the point in a corner
      if ((x0 == xj) && (x1 == yj)) {
        num_crossings = 1;
        break;
      }
      if ((yj > x1) != (yi > x1)) {
        REAL determinate = (x0 - xj) * (yi - yj) - (xi - xj) * (x1 - yj);
        if (determinate == 0) {
          // Point is on line
          num_crossings = 1;
          break;
        }
        if ((determinate < 0) != (yi < yj)) {
          num_crossings++;
        }
      }
    }

    // odd number of crossings implies the point is contained
    if ((num_crossings % 2) == 1) {
      contained = true;
    };
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                          &tmp, &vertices));

    return contained;
  }

  /**
   * @returns The number of labels in the DMPlex.
   */
  inline PetscInt get_num_labels() {
    PetscInt num_labels;
    PETSCCHK(DMGetNumLabels(this->dm, &num_labels));
    return num_labels;
  }

  /**
   * @returns The DMLabel for "Face Sets"
   */
  inline DMLabel get_face_sets_label() {
    const char *name = "Face Sets";
    PetscBool has_label;
    PETSCCHK(DMHasLabel(this->dm, name, &has_label));
    NESOASSERT(has_label,
               "The Face Sets label does not exist on this DMPlex. If using "
               "gmsh, check Physical Lines and Physical Surfaces are set.");
    DMLabel label;
    PETSCCHK(DMGetLabel(this->dm, name, &label));
    return label;
  }

  /**
   * @param index Index of label to get name of.
   * @returns The label name corresponding to an index.
   */
  inline std::string get_label_name(const PetscInt index) {
    const char *name;
    PETSCCHK(DMGetLabelName(this->dm, index, &name));
    return std::string(name);
  }

  /**
   * Get boundary points start and end. In 2D the boundary point types are
   * edges. In 3D the boundary point types are faces. Returns [start, end).
   *
   * @param[in, out] start First boundary point.
   * @param[in, out] end Last boundary point plus one.
   */
  inline void get_boundary_stratum(PetscInt *start, PetscInt *end) {
    PetscInt depth = this->ndim - 1;
    PETSCCHK(DMPlexGetDepthStratum(this->dm, depth, start, end));
  }

  /**
   * @returns Map from face sets int label to DMPlex points with that label.
   */
  inline std::map<PetscInt, std::vector<PetscInt>> get_face_sets() {
    DMLabel face_sets_label = this->get_face_sets_label();
    PetscInt points_start, points_end;
    this->get_boundary_stratum(&points_start, &points_end);

    std::map<PetscInt, std::vector<PetscInt>> map;
    for (PetscInt px = points_start; px < points_end; px++) {
      PetscInt value;
      PETSCCHK(DMLabelGetValue(face_sets_label, px, &value));
      map[value].push_back(px);
    }

    return map;
  }

  /**
   * Write the DM to a file for visualisation in paraview.
   *
   * @param filename Filename for VTK file.
   */
  inline void write_vtk(const std::string filename) {
    PetscViewer viewer;
    PETSCCHK(PetscViewerCreate(PETSC_COMM_SELF, &viewer));
    PETSCCHK(PetscViewerSetType(viewer, PETSCVIEWERVTK));
    PETSCCHK(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
    PETSCCHK(PetscViewerFileSetName(viewer, filename.c_str()));
    PETSCCHK(DMView(this->dm, viewer));
    PETSCCHK(PetscViewerDestroy(&viewer));
  }

  /**
   * Print to stdout information about the held DMPlex.
   */
  inline void print() {
    for (int cx = 0; cx < this->ncells; cx++) {
      const PetscInt point_index = this->map_np_to_petsc.at(cx);
      nprint("---------------------------------------------------------------");
      nprint("Local index:", cx, "point index:", point_index);
      PetscBool is_dg;
      PetscInt nc;
      const PetscScalar *array;
      PetscScalar *coords;
      PETSCCHK(DMPlexGetCellCoordinates(dm, cx, &is_dg, &nc, &array, &coords));
      for (int px = 0; px < nc; px += this->ndim) {
        std::cout << "\t";
        for (int dx = 0; dx < this->ndim; dx++) {
          std::cout << coords[px + dx] << " ";
        }
        std::cout << std::endl;
      }
      PETSCCHK(
          DMPlexRestoreCellCoordinates(dm, cx, &is_dg, &nc, &array, &coords));
    }
  }

  /**
   * Get the volume of a cell in the local mesh.
   *
   * @param cell Local index of cell.
   * @returns Volume of cell.
   */
  inline REAL get_cell_volume(const int index) {
    this->check_valid_local_cell(index);
    const PetscInt petsc_index = this->map_np_to_petsc.at(index);
    PetscReal vol;
    PetscReal centroid[3];
    PetscReal normal[3];
    PETSCCHK(DMPlexComputeCellGeometryFVM(this->dm, petsc_index, &vol, centroid,
                                          normal));
    return vol;
  }

  /**
   * @returns The total volume of the mesh. Must be called collectively on the
   * communicator.
   */
  inline REAL get_volume() {
    if (this->volume < 0.0) {
      double local_volume = 0.0;
      for (int cx = 0; cx < this->ncells; cx++) {
        local_volume += this->get_cell_volume(cx);
      }
      MPICHK(MPI_Allreduce(&local_volume, &this->volume, 1, MPI_DOUBLE, MPI_SUM,
                           this->comm));
    }
    return this->volume;
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
