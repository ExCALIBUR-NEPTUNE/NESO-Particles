#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/dmplex_helper.hpp>

namespace NESO::Particles::PetscInterface {

void generic_distribute(DM *dm, MPI_Comm comm, const PetscInt overlap,
                        PetscSF *sf) {
  int size;
  MPICHK(MPI_Comm_size(comm, &size));
  if (size > 1) {
    DM dm_out;
    PETSCCHK(DMPlexDistribute(*dm, overlap, sf, &dm_out));
    NESOASSERT(dm_out, "Could not distribute mesh.");
    PETSCCHK(DMDestroy(dm));
    *dm = dm_out;
  }
}

void setup_coordinate_section(DM &dm, const PetscInt vertex_start,
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

void setup_local_coordinate_vector(DM &dm, Vec &coordinates) {
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

std::vector<PetscInt> get_global_distributed_points_map(DM &dm_distributed,
                                                        PetscSF &sf) {

  MPI_Comm comm;
  PETSCCHK(PetscObjectGetComm((PetscObject)dm_distributed, &comm));

  int size;
  MPICHK(MPI_Comm_size(comm, &size));

  PetscInt point_start = 0;
  PetscInt point_end = 0;
  PETSCCHK(DMPlexGetChart(dm_distributed, &point_start, &point_end));

  if (size == 1) {
    const PetscInt npoints_global = point_end - point_start;
    std::vector<PetscInt> global_points(npoints_global);
    std::iota(global_points.begin(), global_points.end(), 0);
    return global_points;
  } else {

    PetscInt nroots = 0;
    PetscInt nleaves = 0;
    PetscInt const *ilocal = nullptr;
    PetscSFNode const *iremote = nullptr;
    PETSCCHK(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));

    IS global_point_numbers;
    PETSCCHK(DMPlexCreatePointNumbering(dm_distributed, &global_point_numbers));
    const PetscInt *ptr;
    PETSCCHK(ISGetIndices(global_point_numbers, &ptr));

    PetscInt npoints_global = 0;

    MPICHK(
        MPI_Allreduce(&nleaves, &npoints_global, 1, MPIU_INT, MPI_SUM, comm));

    std::vector<PetscInt> local_points(npoints_global);
    std::vector<PetscInt> global_points(npoints_global);
    std::fill(local_points.begin(), local_points.end(), -1);
    std::fill(global_points.begin(), global_points.end(), -1);

    for (int ix = 0; ix < nleaves; ix++) {

      const PetscInt global_point_previous = iremote[ix].index;
      const PetscInt global_point_current = ptr[ix - point_start];

      NESOASSERT((0 <= global_point_previous) &&
                     (global_point_previous < npoints_global),
                 "Bad global point previous.");

      if (global_point_current > -1) {
        NESOASSERT((global_point_current < npoints_global),
                   "Bad global point current.");

        local_points[global_point_previous] = global_point_current;
      }
    }

    MPICHK(MPI_Allreduce(local_points.data(), global_points.data(),
                         static_cast<int>(npoints_global), MPIU_INT, MPI_MAX,
                         comm));

    PETSCCHK(ISRestoreIndices(global_point_numbers, &ptr));
    PETSCCHK(ISDestroy(&global_point_numbers));

    return global_points;
  }
}

bool dm_from_serialised_cells(
    std::list<DMPlexCellSerialise> &serialised_cells, DM &dm_prototype, DM &dm,
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
      for (auto &point_spec : std_cell.point_cones) {
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
      for (auto &point_spec : std_cell.point_cones) {
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
          PETSCCHK(DMPlexSetConeOrientation(
              dm, local_point,
              std_cell.point_cone_orientations.at(global_point).data()));
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

DMPlexCellSerialise
DMPlexHelper::get_copyable_cell(const PetscInt local_index) {
  this->check_valid_local_cell(local_index);
  const PetscInt point = this->map_np_to_petsc.at(local_index);

  int rank;
  MPICHK(MPI_Comm_rank(this->comm, &rank));
  DMPolytopeType cell_type;
  PETSCCHK(DMPlexGetCellType(dm, point, &cell_type));

  auto lambda_rename = [&](PetscInt cell) -> PetscInt {
    return this->get_point_global_index(cell);
  };
  std::function<PetscInt(PetscInt)> rename_function = lambda_rename;
  auto spec =
      PetscInterface::CellSTDRepresentation(this->dm, point, rename_function);

  std::vector<std::byte> cell_representation;
  spec.serialise(cell_representation);

  DMPlexCellSerialise cs{local_index, lambda_rename(point), rank, cell_type,
                         cell_representation};

  return cs;
}

void DMPlexHelper::free() { PETSCCHK(ISDestroy(&this->global_point_numbers)); }

DMPlexHelper::DMPlexHelper(MPI_Comm comm, DM dm)
    : volume(-1.0), bounding_box(nullptr), comm(comm), dm(dm) {
  DMPlexInterpolatedFlag interpolated;
  PETSCCHK(DMPlexIsInterpolated(this->dm, &interpolated));
  NESOASSERT(interpolated == DMPLEX_INTERPOLATED_FULL,
             "Expected fully interpolated mesh.");
  PETSCCHK(DMGetCoordinateDim(this->dm, &this->ndim));
  PETSCCHK(
      DMPlexGetHeightStratum(this->dm, 0, &this->cell_start, &this->cell_end));
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

  NESOASSERT(static_cast<std::size_t>(ix) == this->map_petsc_to_np.size(),
             "Size missmatch.");
  NESOASSERT(static_cast<std::size_t>(ix) == this->map_np_to_petsc.size(),
             "Size missmatch.");
  this->ncells = this->map_np_to_petsc.size();
  NESOASSERT(this->ncells, "A rank has zero cells.");

  PetscInt point_start = 0;
  PetscInt point_end = 0;
  PETSCCHK(DMPlexGetChart(dm, &point_start, &point_end));

  for (PetscInt px = point_start; px < point_end; px++) {
    const PetscInt global_point = signed_global_id_to_global_id(
        this->internal_get_point_global_index(px));
    this->map_gobal_point_to_local_point[global_point] = px;
  }
}

int DMPlexHelper::get_cell_count() { return this->ncells; }

int DMPlexHelper::get_global_cell_count() {

  if (this->ncells_global < 0) {
    int ncells_local = this->get_cell_count();
    int tmp = -1;
    MPICHK(MPI_Allreduce(&ncells_local, &tmp, 1, MPI_INT, MPI_SUM, this->comm));
    this->ncells_global = tmp;
  }

  return this->ncells_global;
}

PetscInt DMPlexHelper::get_dmplex_cell_index(const PetscInt local_index) {
  this->check_valid_local_cell(local_index);
  const auto index = this->map_np_to_petsc.at(local_index);
  this->check_valid_petsc_cell(index);
  return index;
}

PetscInt DMPlexHelper::get_local_cell_index(const PetscInt petsc_index) {
  this->check_valid_petsc_cell(petsc_index);
  const auto index = this->map_petsc_to_np.at(petsc_index);
  this->check_valid_local_cell(index);
  return index;
}

PetscInt DMPlexHelper::signed_global_id_to_global_id(const PetscInt c) {
  return (c > -1) ? c : ((c * (-1)) - 1);
}

PetscInt DMPlexHelper::get_point_global_index(const PetscInt point,
                                              const bool signed_point) {
  NESOASSERT((this->point_start <= point) && (point < this->point_end),
             "Bad point passed.");
  PetscInt global_point = this->internal_get_point_global_index(point);

  if (signed_point) {
    return global_point;
  } else {
    return signed_global_id_to_global_id(global_point);
  }
}

PetscInt DMPlexHelper::get_local_point_from_global_point(
    const PetscInt global_point_index) {
  NESOASSERT(this->map_gobal_point_to_local_point.count(
                 signed_global_id_to_global_id(global_point_index)),
             "Global point not found.");
  return this->map_gobal_point_to_local_point[global_point_index];
}

ExternalCommon::BoundingBoxSharedPtr DMPlexHelper::get_bounding_box() {
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

ExternalCommon::BoundingBoxSharedPtr
DMPlexHelper::get_cell_bounding_box(const PetscInt cell) {
  this->check_valid_local_cell(cell);

  std::vector<REAL> bb = {
      std::numeric_limits<REAL>::max(),    std::numeric_limits<REAL>::max(),
      std::numeric_limits<REAL>::max(),    std::numeric_limits<REAL>::lowest(),
      std::numeric_limits<REAL>::lowest(), std::numeric_limits<REAL>::lowest()};

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

void DMPlexHelper::get_generic_vertices(
    const PetscInt petsc_index, std::vector<std::vector<REAL>> &vertices) {

  const PetscScalar *array;
  PetscScalar *coords = nullptr;
  PetscInt num_coords;
  PetscBool is_dg;
  this->check_valid_petsc_point(petsc_index);
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

void DMPlexHelper::get_cell_vertices(const PetscInt cell,
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

void DMPlexHelper::get_cell_vertex_average(const PetscInt cell,
                                           std::vector<REAL> &average) {
  this->check_valid_local_cell(cell);
  NESOASSERT(average.size() == static_cast<std::size_t>(this->ndim),
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

int DMPlexHelper::contains_point(std::vector<PetscScalar> &point) {
  const PetscInt ndim = this->ndim;
  NESOASSERT(point.size() == static_cast<std::size_t>(ndim),
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

bool DMPlexHelper::cell_contains_point(const PetscInt index,
                                       std::vector<PetscScalar> &point) {
  const PetscInt ndim = this->ndim;
  NESOASSERT(point.size() == static_cast<std::size_t>(ndim),
             "Miss-match in point size and mesh dimension.");
  NESOASSERT(2 == ndim, "Only implemented in 2D.");
  this->check_valid_local_cell(index);
  const PetscInt petsc_index = this->map_np_to_petsc.at(index);
  bool contained = false;
  const PetscScalar x0 = point.at(0);
  const PetscScalar x1 = point.at(1);

  const PetscScalar *tmp;
  PetscScalar *vertices = nullptr;
  PetscInt num_crossings = 0, num_coords;
  PetscBool is_dg;

  PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords, &tmp,
                                    &vertices));
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
    const REAL xi_t = vertices[faces[2 * facex + 0] * 2 + 0];
    const REAL yi_t = vertices[faces[2 * facex + 0] * 2 + 1];
    const REAL xj_t = vertices[faces[2 * facex + 1] * 2 + 0];
    const REAL yj_t = vertices[faces[2 * facex + 1] * 2 + 1];

    REAL xi = 0.0;
    REAL yi = 0.0;
    REAL xj = 0.0;
    REAL yj = 0.0;

    consistent_line_orientation_2d(xi_t, yi_t, xj_t, yj_t, &xi, &yi, &xj, &yj);

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

PetscInt DMPlexHelper::get_num_labels() {
  PetscInt num_labels;
  PETSCCHK(DMGetNumLabels(this->dm, &num_labels));
  return num_labels;
}

DMLabel DMPlexHelper::get_face_sets_label() {
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

std::string DMPlexHelper::get_label_name(const PetscInt index) {
  const char *name;
  PETSCCHK(DMGetLabelName(this->dm, index, &name));
  return std::string(name);
}

void DMPlexHelper::get_boundary_stratum(PetscInt *start, PetscInt *end) {
  PetscInt depth = this->ndim - 1;
  PETSCCHK(DMPlexGetDepthStratum(this->dm, depth, start, end));
}

std::map<PetscInt, std::vector<PetscInt>> DMPlexHelper::get_face_sets() {
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

void DMPlexHelper::write_vtk(const std::string filename) {
  PetscViewer viewer;
  PETSCCHK(PetscViewerCreate(PETSC_COMM_SELF, &viewer));
  PETSCCHK(PetscViewerSetType(viewer, PETSCVIEWERVTK));
  PETSCCHK(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
  PETSCCHK(PetscViewerFileSetName(viewer, filename.c_str()));
  PETSCCHK(DMView(this->dm, viewer));
  PETSCCHK(PetscViewerDestroy(&viewer));
}

std::vector<VTK::UnstructuredCell> DMPlexHelper::get_vtk_cell_data() {
  const int cell_count = this->get_cell_count();
  std::vector<VTK::UnstructuredCell> data(cell_count);
  std::vector<std::vector<REAL>> vertices;
  NESOASSERT(this->ndim == 2, "Only implemented in 2D.");
  for (int cellx = 0; cellx < cell_count; cellx++) {
    vertices.clear();
    this->get_cell_vertices(cellx, vertices);
    const int num_vertices = vertices.size();
    data.at(cellx).num_points = num_vertices;
    data.at(cellx).cell_type = num_vertices == 3 ? VTK::CellType::triangle
                                                 : VTK::CellType::quadrilateral;
    data.at(cellx).points.reserve(num_vertices * 3);
    for (int vx = 0; vx < num_vertices; vx++) {
      for (int dx = 0; dx < this->ndim; dx++) {
        data.at(cellx).points.push_back(vertices.at(vx).at(dx));
      }
      for (int dx = this->ndim; dx < 3; dx++) {
        data.at(cellx).points.push_back(0.0);
      }
    }
  }
  return data;
}

void DMPlexHelper::print() {
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

REAL DMPlexHelper::get_cell_volume(const int index) {
  this->check_valid_local_cell(index);
  const PetscInt petsc_index = this->map_np_to_petsc.at(index);
  PetscReal vol;
  PetscReal centroid[3];
  PetscReal normal[3];
  PETSCCHK(DMPlexComputeCellGeometryFVM(this->dm, petsc_index, &vol, centroid,
                                        normal));
  return vol;
}

REAL DMPlexHelper::get_volume() {
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

std::tuple<std::shared_ptr<CellDatConst<int>>,
           std::shared_ptr<CellDatConst<REAL>>>
get_cell_vertices_cdc(SYCLTargetSharedPtr sycl_target,
                      std::shared_ptr<DMPlexHelper> dmh) {
  std::tuple<std::shared_ptr<CellDatConst<int>>,
             std::shared_ptr<CellDatConst<REAL>>>
      d;
  const int cell_count = dmh->get_cell_count();

  std::vector<std::vector<std::vector<REAL>>> vertices(cell_count);

  std::size_t max_num_vertices = 0;
  const auto ndim = dmh->ndim;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    dmh->get_cell_vertices(cellx, vertices.at(cellx));
    max_num_vertices = std::max(max_num_vertices, vertices.at(cellx).size());
  }

  std::get<0>(d) =
      std::make_shared<CellDatConst<int>>(sycl_target, cell_count, 1, 1);
  std::get<1>(d) = std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count,
                                                        max_num_vertices, ndim);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int num_vertices = vertices.at(cellx).size();
    std::get<0>(d)->set_value(cellx, 0, 0, num_vertices);
    auto c = std::get<1>(d)->get_cell(cellx);
    for (int rx = 0; rx < num_vertices; rx++) {
      for (int cx = 0; cx < ndim; cx++) {
        c->at(rx, cx) = vertices.at(cellx).at(rx).at(cx);
      }
    }
    std::get<1>(d)->set_cell(cellx, c);
  }

  return d;
}

std::pair<int, std::vector<int>>
get_map_from_global_cell_points_to_ranks(DM dm) {

  MPI_Comm comm = MPI_COMM_NULL;
  PETSCCHK(PetscObjectGetComm((PetscObject)dm, &comm));

  DMPlexHelper dmh(comm, dm);
  const int global_cell_count = dmh.get_global_cell_count();
  const int cell_count = dmh.get_cell_count();

  std::vector<int> cell_owners_local(global_cell_count);
  std::fill(cell_owners_local.begin(), cell_owners_local.end(), -1);

  int point_min = std::numeric_limits<int>::max();
  int point_max = std::numeric_limits<int>::lowest();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    // local point index of the cell
    const PetscInt point_index = dmh.get_dmplex_cell_index(cellx);
    const int global_point_index =
        static_cast<int>(dmh.get_point_global_index(point_index));
    point_min = std::min(point_min, global_point_index);
    point_max = std::max(point_max, global_point_index);
  }

  int global_point_min = 0;
  int global_point_max = 0;

  MPICHK(
      MPI_Allreduce(&point_min, &global_point_min, 1, MPI_INT, MPI_MIN, comm));
  MPICHK(
      MPI_Allreduce(&point_max, &global_point_max, 1, MPI_INT, MPI_MAX, comm));
  NESOASSERT(global_point_max - global_point_min + 1 == global_cell_count,
             "Error deducing petsc numbering");

  int rank = 0;
  MPICHK(MPI_Comm_rank(comm, &rank));

  for (int cellx = 0; cellx < cell_count; cellx++) {
    // local point index of the cell
    const PetscInt point_index = dmh.get_dmplex_cell_index(cellx);
    const int global_point_index =
        static_cast<int>(dmh.get_point_global_index(point_index));
    const int index = global_point_index - global_point_min;
    cell_owners_local.at(index) = rank;
  }

  std::vector<int> cell_owners(global_cell_count);
  MPICHK(MPI_Allreduce(cell_owners_local.data(), cell_owners.data(),
                       global_cell_count, MPI_INT, MPI_MAX, comm));
  cell_owners_local.clear();

  return {global_point_min, cell_owners};
}

} // namespace NESO::Particles::PetscInterface

#endif
