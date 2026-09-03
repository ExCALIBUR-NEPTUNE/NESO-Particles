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

void print_transitive_closure(DM &dm, const PetscInt point) {
  PetscInt num_points = 0;
  PetscInt *points = NULL;
  PetscInt depth = 0;

  PETSCCHK(
      DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &num_points, &points));

  nprint("transitive closure for point:", point);

  for (PetscInt px = 0; px < num_points; px++) {
    const PetscInt pointx = points[2 * px];
    const PetscInt orientation = points[2 * px + 1];
    PETSCCHK(DMPlexGetPointDepth(dm, pointx, &depth));

    nprint("\tcone index:", px, "point:", pointx, "orientation:", orientation,
           "depth:", depth);
  }

  PETSCCHK(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &num_points,
                                          &points));
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

VTK::CellType get_vtk_cell_type(const DMPolytopeType petsc_cell_type) {
  switch (petsc_cell_type) {
  case DM_POLYTOPE_POINT:
    return VTK::CellType::point;
  case DM_POLYTOPE_SEGMENT:
    return VTK::CellType::line;
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
    return VTK::CellType::line;
  case DM_POLYTOPE_TRIANGLE:
    return VTK::CellType::triangle;
  case DM_POLYTOPE_QUADRILATERAL:
    return VTK::CellType::quadrilateral;
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
    return VTK::CellType::quadrilateral;
  case DM_POLYTOPE_TETRAHEDRON:
    return VTK::CellType::tetrahedron;
  case DM_POLYTOPE_PYRAMID:
    return VTK::CellType::pyramid;
  case DM_POLYTOPE_TRI_PRISM:
    return VTK::CellType::wedge;
  case DM_POLYTOPE_TRI_PRISM_TENSOR:
    return VTK::CellType::wedge;
  case DM_POLYTOPE_HEXAHEDRON:
    return VTK::CellType::hex;
  case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    return VTK::CellType::hex;
  default:
    NESOASSERT(false, "Unknown point type: " + std::to_string(petsc_cell_type));
    return VTK::CellType::point;
  }
}

void split_quadrilateral_into_two_triangles(
    DM dm, const PetscInt quad_index,
    std::array<std::array<PetscInt, 3>, 2> &triangle_indices) {

  PetscInt start = 0, end = 0;
  PETSCCHK(DMPlexGetDepthStratum(dm, 2, &start, &end));
  NESOASSERT(start <= quad_index && quad_index < end,
             "Invalid index for quadrilateral.");

  PetscInt cone_size = 0;
  PETSCCHK(DMPlexGetConeSize(dm, quad_index, &cone_size));

  NESOASSERT(cone_size == 4,
             "Expected quadrilateral to have a cone size of four.");

  const PetscInt *cone = nullptr;
  PETSCCHK(DMPlexGetCone(dm, quad_index, &cone));

  // Get the vertices from the quad in order.
  const PetscInt *edge_cone = nullptr;

  std::array<PetscInt, 4> vertices;
  std::map<PetscInt, std::set<PetscInt>> map_vertex_to_neighbours;

  PetscInt first_vertex = -1;
  for (int edgex = 0; edgex < 4; edgex++) {
    const PetscInt edge_index = cone[edgex];
    PETSCCHK(DMPlexGetCone(dm, edge_index, &edge_cone));
    const auto v0 = edge_cone[0];
    const auto v1 = edge_cone[1];

    if (edgex == 0) {
      first_vertex = v0;
    }

    map_vertex_to_neighbours[v0].insert(v1);
    map_vertex_to_neighbours[v1].insert(v0);
  }

  PetscInt current_vertex = first_vertex;
  for (int edgex = 0; edgex < 4; edgex++) {

    vertices.at(edgex) = current_vertex;
    // get a neighbour vertex
    const PetscInt next_vertex =
        *map_vertex_to_neighbours.at(current_vertex).begin();
    // Remove the current point from the neighbours of the next point such that
    // the loop never travels backwards.
    map_vertex_to_neighbours.at(next_vertex).erase(current_vertex);

    current_vertex = next_vertex;
  }

  // For the four vertices [0,1,2,3] there are two possible splits. 1) the new
  // edge [0,2] or 2) the new edge [1,3]. We compare the lengths of these two
  // new possible edges and choose the shortest edge.

  auto lambda_get_coords = [&](const PetscInt petsc_index) {
    const PetscScalar *array;
    PetscScalar *coords = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;
    PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                      &array, &coords));
    NESOASSERT(coords != nullptr, "No vertices returned for cell.");
    NESOASSERT(num_coords == 3 || num_coords == 2,
               "Expected two or three coordinates.");

    std::vector<PetscScalar> verticest(num_coords);
    for (int dx = 0; dx < num_coords; dx++) {
      verticest.at(dx) = coords[dx];
    }

    PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                          &array, &coords));

    return verticest;
  };

  auto lambda_distance2 = [&](const auto &a, const auto &b) {
    const std::size_t N = a.size();
    PetscScalar d2 = 0.0;
    for (std::size_t ix = 0; ix < N; ix++) {
      const auto diff = a.at(ix) - b.at(ix);
      d2 += diff * diff;
    }
    return d2;
  };

  auto coords0 = lambda_get_coords(vertices.at(0));
  auto coords1 = lambda_get_coords(vertices.at(1));
  auto coords2 = lambda_get_coords(vertices.at(2));
  auto coords3 = lambda_get_coords(vertices.at(3));

  const PetscScalar distance_02 = lambda_distance2(coords0, coords1);
  const PetscScalar distance_13 = lambda_distance2(coords1, coords3);

  if (distance_02 <= distance_13) {
    triangle_indices.at(0) = {vertices.at(0), vertices.at(1), vertices.at(2)};
    triangle_indices.at(1) = {vertices.at(0), vertices.at(2), vertices.at(3)};
  } else {
    triangle_indices.at(0) = {vertices.at(0), vertices.at(1), vertices.at(3)};
    triangle_indices.at(1) = {vertices.at(1), vertices.at(2), vertices.at(3)};
  }
}

void HaloDMIndexMapper::get_depth_stratum(const PetscInt depth, PetscInt *start,
                                          PetscInt *end) {
  *start = this->depth_starts.at(depth);
  *end = this->depth_ends.at(depth);
}

PetscInt HaloDMIndexMapper::get_local_point_index(const PetscInt point) {
  const auto local_point = this->map_global_to_local.at(point);
  return local_point;
}

HaloDMIndexMapper::HaloDMIndexMapper(
    std::vector<CellSTDRepresentation> &cells) {
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

namespace {
bool std_cell_representation_is_self_consistent(
    std::vector<CellSTDRepresentation> &std_rep_cells) {

  bool v = true;

  std::map<PetscInt, std::vector<PetscInt>> map_point_to_cone;

  for (auto &std_cell : std_rep_cells) {
    for (auto &point_spec : std_cell.point_cones) {
      const PetscInt global_point = point_spec.first;
      const auto &cone = point_spec.second;

      if (map_point_to_cone.count(global_point)) {
        bool same = cone.size() == map_point_to_cone[global_point].size();
        if (same) {
          for (int ix = 0; ix < cone.size(); ix++) {
            if (same) {
              same = cone[ix] == map_point_to_cone[global_point][ix];
            }
          }
        }

        if (!same) {
          v = false;
        }
      } else {
        map_point_to_cone[global_point] = cone;
      }
    }
  }

  return v;
}

bool std_cell_representation_matches_dm(
    std::vector<CellSTDRepresentation> &std_rep_cells,
    HaloDMIndexMapper &index_mapper, DM dm) {
  bool v = true;

  std::set<PetscInt> checked_global_points;
  std::vector<PetscInt> cone_local;

  for (auto &std_cell : std_rep_cells) {
    for (auto &point_spec : std_cell.point_cones) {
      const PetscInt global_point = point_spec.first;
      if (!checked_global_points.count(global_point)) {
        checked_global_points.insert(global_point);
        const PetscInt local_point =
            index_mapper.get_local_point_index(global_point);

        cone_local.clear();
        auto &cone_global = point_spec.second;
        int lx = 0;
        for (auto gx : cone_global) {
          cone_local.push_back(index_mapper.get_local_point_index(gx));
          lx++;
        }

        const PetscInt *cone = nullptr;
        const PetscInt *ornt = nullptr;
        PETSCCHK(DMPlexGetOrientedCone(dm, local_point, &cone, &ornt));
        PetscInt cone_size = 0;
        PETSCCHK(DMPlexGetConeSize(dm, local_point, &cone_size));

        const bool size_matches =
            cone_size ==
            std_cell.point_cone_orientations.at(global_point).size();

        bool entries_match = true;

        if (size_matches) {
          for (PetscInt ix = 0; ix < cone_size; ix++) {
            const bool entry_matches =
                (cone_local.at(ix) == cone[ix]) &&
                (std_cell.point_cone_orientations.at(global_point).at(ix) ==
                 ornt[ix]);
            if (!entry_matches) {
              entries_match = false;
            }
          }
        }

        PETSCCHK(DMPlexRestoreOrientedCone(dm, local_point, &cone, &ornt));

        if (!(entries_match && size_matches)) {
          v = false;
        }
      }
    }
  }

  return v;
}
} // namespace

bool dm_from_serialised_cells(
    std::list<DMPlexCellSerialise> &serialised_cells, DM &dm_prototype, DM &dm,
    std::map<PetscInt, std::tuple<int, PetscInt, PetscInt>>
        &map_local_lid_remote_lid,
    const bool additional_checks) {

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
    NESOASSERT((!additional_checks) ||
                   std_cell_representation_is_self_consistent(std_rep_cells),
               "std_cell_representation_is_self_consistent check failed.");

    PETSCCHK(DMSetUp(dm));
    std::vector<PetscInt> cone_local;
    points_set.clear();

    PetscInt prototype_depth = 0;
    PETSCCHK(DMPlexGetDepth(dm_prototype, &prototype_depth));

    for (PetscInt depthx = 0; depthx <= prototype_depth; depthx++) {
      for (auto &std_cell : std_rep_cells) {
        for (auto &point_spec : std_cell.point_cones) {
          const PetscInt global_point = point_spec.first;

          if (std_cell.get_point_depth(global_point) == depthx) {

            if (!points_set.count(global_point)) {
              points_set.insert(global_point);
              const PetscInt local_point =
                  index_mapper.get_local_point_index(global_point);

              auto &cone_global = point_spec.second;
              cone_local.clear();

              int lx = 0;
              for (auto gx : cone_global) {
                cone_local.push_back(index_mapper.get_local_point_index(gx));
                lx++;
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

  NESOASSERT((!additional_checks) || std_cell_representation_matches_dm(
                                         std_rep_cells, index_mapper, dm),
             "std_cell_representation_matches_dm check failed.");

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

  // Get the bounds of global indices on the faces
  {
    PetscInt local_start = 0;
    PetscInt local_end = 0;
    this->get_boundary_stratum(&local_start, &local_end);

    PetscInt l = std::numeric_limits<PetscInt>::max();
    PetscInt u = std::numeric_limits<PetscInt>::lowest();

    for (PetscInt localx = local_start; localx < local_end; localx++) {
      const PetscInt global_point_index = this->signed_global_id_to_global_id(
          this->internal_get_point_global_index(localx));
      l = std::min(l, global_point_index);
      u = std::max(u, global_point_index);
    }

    INT ll = l;
    INT lu = u;
    INT gl = l;
    INT gu = u;

    MPICHK(MPI_Allreduce(&ll, &gl, 1, map_ctype_mpi_type<INT>(), MPI_MIN,
                         this->comm));
    MPICHK(MPI_Allreduce(&lu, &gu, 1, map_ctype_mpi_type<INT>(), MPI_MAX,
                         this->comm));

    this->boundary_index_bound_lower = gl;
    this->boundary_index_bound_upper = gu + 1;
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
DMPlexHelper::get_point_bounding_box(const PetscInt petsc_index) {

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
  this->check_valid_petsc_point(petsc_index);
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

ExternalCommon::BoundingBoxSharedPtr
DMPlexHelper::get_cell_bounding_box(const PetscInt cell) {
  this->check_valid_local_cell(cell);
  const PetscInt petsc_index = this->map_np_to_petsc.at(cell);
  this->check_valid_petsc_cell(petsc_index);
  return this->get_point_bounding_box(petsc_index);
}

void DMPlexHelper::get_point_vertices(
    const PetscInt petsc_index, std::vector<std::vector<REAL>> &vertices) {

  const PetscInt *o = nullptr;

  PETSCCHK(DMPlexGetConeOrientation(this->dm, petsc_index, &o));

  PetscInt cone_size = 0;
  PETSCCHK(DMPlexGetConeSize(this->dm, petsc_index, &cone_size));

  if (petsc_index == 1848) {
    nprint("get_point_vertices:", petsc_index);
    for (int cx = 0; cx < cone_size; cx++) {
      nprint("\t", o[cx]);
    }

    Vec coordinates;
    PetscScalar *coef = nullptr;
    PetscInt size = 0;

    DM cdm, plex;

    PETSCCHK(DMGetCoordinateDM(dm, &cdm));
    PETSCCHK(DMGetCoordinatesLocal(dm, &coordinates));
    PETSCCHK(DMConvert(cdm, DMPLEX, &plex));

    PETSCCHK(DMGetCoordinatesLocal(dm, &coordinates));
    PETSCCHK(DMPlexVecGetClosure(plex, NULL, coordinates, petsc_index, &size,
                                 &coef));

    nprint_variable(size);
    for (int ix = 0; ix < 4; ix++) {
      nprint(coef[ix * 3 + 0], coef[ix * 3 + 1], coef[ix * 3 + 2]);
    }
    nprint("---_");

    PETSCCHK(DMPlexVecRestoreClosure(plex, NULL, coordinates, petsc_index,
                                     &size, &coef));
  }

  const PetscScalar *array;
  PetscScalar *coords = nullptr;
  PetscInt num_coords;
  PetscBool is_dg;
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

  if (petsc_index == 1848) {
    for (auto vx : vertices) {
      nprint(vx[0], vx[1], vx[2]);
    }
  }
}

void DMPlexHelper::get_cell_vertices(const PetscInt cell,
                                     std::vector<std::vector<REAL>> &vertices) {
  this->check_valid_local_cell(cell);

  const PetscInt petsc_index = this->map_np_to_petsc.at(cell);

  this->check_valid_petsc_cell(petsc_index);
  return this->get_point_vertices(petsc_index, vertices);
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

/**
 * Get the point type.
 *
 * @param point_index Local point index.
 * @returns PETSc description of cell type.
 */
DMPolytopeType DMPlexHelper::get_point_type(const PetscInt point_index) {
  this->check_valid_petsc_point(point_index);
  DMPolytopeType cell_type;
  PETSCCHK(DMPlexGetCellType(this->dm, point_index, &cell_type));
  return cell_type;
}

DMPolytopeType DMPlexHelper::get_cell_type(const PetscInt cell) {
  this->check_valid_local_cell(cell);
  const PetscInt petsc_index = this->map_np_to_petsc.at(cell);
  return this->get_point_type(petsc_index);
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

bool DMPlexHelper::cell_contains_point_2d(const PetscInt index,
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

bool DMPlexHelper::cell_contains_point_3d(const PetscInt index,
                                          std::vector<PetscScalar> &point) {
  NESOASSERT(this->ndim == 3, "Only implemented in 3D.");
  const PetscInt ndim = this->ndim;
  NESOASSERT(point.size() == static_cast<std::size_t>(ndim),
             "Miss-match in point size and mesh dimension.");
  this->check_valid_local_cell(index);
  const PetscInt petsc_index = this->map_np_to_petsc.at(index);

  const PetscScalar x0 = point.at(0);
  const PetscScalar x1 = point.at(1);
  const PetscScalar x2 = point.at(2);

  PetscInt num_faces = 0;
  PETSCCHK(DMPlexGetConeSize(this->dm, petsc_index, &num_faces));

  const PetscScalar *tmp;
  PetscScalar *vertices = nullptr;
  PetscInt num_coords;
  PetscBool is_dg;

  PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords, &tmp,
                                    &vertices));
  const int num_vertices = num_coords / 3;
  std::vector<PetscScalar> h_vertices;
  h_vertices.reserve(num_coords);
  for (PetscInt ix = 0; ix < num_coords; ix++) {
    h_vertices.push_back(vertices[ix]);
  }
  PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                        &tmp, &vertices));

  const PetscInt *cone = nullptr;
  PETSCCHK(DMPlexGetCone(this->dm, petsc_index, &cone));

  bool below_all_planes = true;

  for (PetscInt facex = 0; facex < num_faces; facex++) {
    const PetscInt face_petsc_index = cone[facex];

    PETSCCHK(DMPlexGetCellCoordinates(dm, face_petsc_index, &is_dg, &num_coords,
                                      &tmp, &vertices));
    NESOASSERT(num_coords == 12 || num_coords == 9, "Unexpected num coords.");
    const PetscInt c0 = 0;
    const PetscInt c1 = 1;
    const PetscInt c2 = num_coords == 12 ? 3 : 2;

    const PetscScalar v0[3] = {vertices[3 * c1 + 0] - vertices[3 * c0 + 0],
                               vertices[3 * c1 + 1] - vertices[3 * c0 + 1],
                               vertices[3 * c1 + 2] - vertices[3 * c0 + 2]};

    const PetscScalar v1[3] = {vertices[3 * c2 + 0] - vertices[3 * c0 + 0],
                               vertices[3 * c2 + 1] - vertices[3 * c0 + 1],
                               vertices[3 * c2 + 2] - vertices[3 * c0 + 2]};

    PetscScalar n[3] = {0.0, 0.0, 0.0};
    KERNEL_CROSS_PRODUCT_3D(v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], n[0],
                            n[1], n[2]);

    int direction = 0;

    for (PetscInt vx = 0; vx < num_vertices; vx++) {

      // vector from first vertex to test point
      const PetscScalar t0[3] = {h_vertices.at(3 * vx + 0) - vertices[0],
                                 h_vertices.at(3 * vx + 1) - vertices[1],
                                 h_vertices.at(3 * vx + 2) - vertices[2]};

      const PetscScalar t0_dot_n =
          KERNEL_DOT_PRODUCT_3D(t0[0], t0[1], t0[2], n[0], n[1], n[2]);

      if (std::fabs(t0_dot_n) > 1.0e-6) {
        const int to_test_direction = t0_dot_n > 0.0 ? 1 : -1;
        if (direction == 0) {
          direction = to_test_direction;
        } else {
          NESOASSERT(direction == to_test_direction,
                     "Inconsistent directions.");
        }
      }
    }

    NESOASSERT(direction != 0, "Could not determine direction");
    // normal points inwards
    if (direction > 0) {
      n[0] *= -1;
      n[1] *= -1;
      n[2] *= -1;
    }

    const REAL normalisation =
        1.0 / std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

    n[0] *= normalisation;
    n[1] *= normalisation;
    n[2] *= normalisation;

    // Now normal point outwards
    const PetscScalar t1[3] = {x0 - vertices[0], x1 - vertices[1],
                               x2 - vertices[2]};

    const PetscScalar t0_dot_n =
        KERNEL_DOT_PRODUCT_3D(t1[0], t1[1], t1[2], n[0], n[1], n[2]);

    if (t0_dot_n > 0.0) {
      below_all_planes = false;
    }

    PETSCCHK(DMPlexRestoreCellCoordinates(dm, face_petsc_index, &is_dg,
                                          &num_coords, &tmp, &vertices));
  }

  return below_all_planes;
}

bool DMPlexHelper::cell_contains_point(const PetscInt index,
                                       std::vector<PetscScalar> &point) {

  NESOASSERT(this->ndim != 1, "Only implemented in 2D and 3D.");
  if (this->ndim == 2) {
    return this->cell_contains_point_2d(index, point);
  } else {
    return this->cell_contains_point_3d(index, point);
  }
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

  INT bound_lower = 0;
  INT bound_upper = 0;
  this->get_global_face_index_bounds(bound_lower, bound_upper);

  std::map<PetscInt, std::vector<PetscInt>> map;
  for (PetscInt px = points_start; px < points_end; px++) {

    // Only return facets which this rank owns for the case when the face label
    // includes internal faces.
    const PetscInt global_index = internal_get_point_global_index(px);
    if (global_index >= 0) {
      NESOASSERT((bound_lower <= global_index) && (global_index < bound_upper),
                 "Bad global point or badly computed global bounds.");
      PetscInt value;
      PETSCCHK(DMLabelGetValue(face_sets_label, px, &value));
      map[value].push_back(px);
    }
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

const VTK::UnstructuredCell &
DMPlexHelper::get_vtk_point_data(const PetscInt index) {

  this->check_valid_petsc_point(index);
  std::vector<std::vector<REAL>> vertices;
  std::vector<PetscInt> order;
  VTK::UnstructuredCell data;

  if (this->map_petsc_to_vtk.count(index) == 0) {
    this->get_point_vertices(index, vertices);
    const int num_vertices = vertices.size();
    data.num_points = num_vertices;
    const auto cell_type = this->get_point_type(index);
    const auto vtk_cell_type = get_vtk_cell_type(cell_type);

    data.cell_type = vtk_cell_type;
    data.points.reserve(num_vertices * 3);
    this->get_vtk_point_vertex_order(index, order);

    for (int vx = 0; vx < num_vertices; vx++) {
      for (int dx = 0; dx < this->ndim; dx++) {
        data.points.push_back(vertices.at(order.at(vx)).at(dx));
      }
      for (int dx = this->ndim; dx < 3; dx++) {
        data.points.push_back(0.0);
      }
    }
    this->map_petsc_to_vtk[index] = data;
  }

  return this->map_petsc_to_vtk.at(index);
}

std::vector<VTK::UnstructuredCell> DMPlexHelper::get_vtk_cell_data() {
  const int cell_count = this->get_cell_count();
  std::vector<VTK::UnstructuredCell> data;
  data.reserve(cell_count);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const PetscInt petsc_index = this->map_np_to_petsc.at(cellx);
    data.push_back(this->get_vtk_point_data(petsc_index));
  }
  return data;
}

void DMPlexHelper::get_vtk_point_vertex_order(const PetscInt index,
                                              std::vector<PetscInt> &order) {

  this->check_valid_petsc_point(index);
  std::map<VTK::CellType, std::vector<int>> map_shape_to_order;
  map_shape_to_order[VTK::CellType::point] = {0};
  map_shape_to_order[VTK::CellType::line] = {0, 1};
  map_shape_to_order[VTK::CellType::triangle] = {0, 1, 2};
  map_shape_to_order[VTK::CellType::quadrilateral] = {0, 1, 2, 3};
  map_shape_to_order[VTK::CellType::tetrahedron] = {0, 1, 2, 3};
  map_shape_to_order[VTK::CellType::pyramid] = {0, 1, 3, 2, 4};
  map_shape_to_order[VTK::CellType::wedge] = {0, 1, 2, 3, 4, 5};
  map_shape_to_order[VTK::CellType::hex] = {1, 2, 6, 7, 0, 3, 5, 4};

  const auto cell_type = this->get_point_type(index);
  const auto vtk_cell_type = get_vtk_cell_type(cell_type);
  const auto &ref_order = map_shape_to_order.at(vtk_cell_type);

  order.clear();
  order.insert(order.end(), ref_order.begin(), ref_order.end());
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

REAL DMPlexHelper::get_point_volume(const PetscInt point_index) {
  this->check_valid_petsc_point(point_index);
  PetscReal vol;
  PetscReal centroid[3];
  PetscReal normal[3];
  PETSCCHK(DMPlexComputeCellGeometryFVM(this->dm, point_index, &vol, centroid,
                                        normal));
  return vol;
}

REAL DMPlexHelper::get_cell_volume(const int index) {
  this->check_valid_local_cell(index);
  const PetscInt petsc_index = this->map_np_to_petsc.at(index);
  return this->get_point_volume(petsc_index);
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

void DMPlexHelper::get_linear_normal_vector(const PetscInt point_index,
                                            std::vector<REAL> &normal_vector) {

  PetscInt depth = -1;
  PETSCCHK(DMPlexGetPointDepth(dm, point_index, &depth));
  NESOASSERT(depth == 2, "Only implemented for linear 2D faces on 3D meshes.");

  std::vector<std::vector<REAL>> vertices;
  this->get_point_vertices(point_index, vertices);
  NESOASSERT(vertices.size() > 2, "Expected at least two vertices.");

  std::array<REAL, 3> v0 = {vertices.at(0).at(0), vertices.at(0).at(1),
                            vertices.at(0).at(2)};
  std::array<REAL, 3> v1 = {vertices.at(1).at(0), vertices.at(1).at(1),
                            vertices.at(1).at(2)};
  std::array<REAL, 3> v2 = {vertices.at(2).at(0), vertices.at(2).at(1),
                            vertices.at(2).at(2)};

  std::array<REAL, 3> E01 = {0.0, 0.0, 0.0};
  std::array<REAL, 3> E02 = {0.0, 0.0, 0.0};
  for (int dx = 0; dx < 3; dx++) {
    E01[dx] = v1[dx] - v0[dx];
    E02[dx] = v2[dx] - v0[dx];
  }

  normal_vector.clear();
  normal_vector.resize(3);
  KERNEL_CROSS_PRODUCT_3D(E01[0], E01[1], E01[2], E02[0], E02[1], E02[2],
                          normal_vector[0], normal_vector[1], normal_vector[2]);

  // Normalise the vector
  const REAL normal_length2 = KERNEL_DOT_PRODUCT_3D(
      normal_vector[0], normal_vector[1], normal_vector[2], normal_vector[0],
      normal_vector[1], normal_vector[2]);

  const REAL norm_scaling = 1.0 / std::sqrt(normal_length2);
  normal_vector[0] *= norm_scaling;
  normal_vector[1] *= norm_scaling;
  normal_vector[2] *= norm_scaling;

  // Now that we have a normal vector we orientate it to point away from the
  // first element in the support if there is a support.
  PetscInt support_size = -1;
  PETSCCHK(DMPlexGetSupportSize(dm, point_index, &support_size));

  if (support_size > 0) {
    const PetscInt *support = nullptr;
    PETSCCHK(DMPlexGetSupport(dm, point_index, &support));
    const PetscInt point_index_support = support[0];

    this->get_point_vertices(point_index_support, vertices);

    std::array<REAL, 3> average = {0.0, 0.0, 0.0};
    const REAL scaling = 1.0 / vertices.size();
    for (auto &vx : vertices) {
      average[0] += vx.at(0) * scaling;
      average[1] += vx.at(1) * scaling;
      average[2] += vx.at(2) * scaling;
    }

    // Vector from v0 to the test point
    std::array<REAL, 3> Etest = {average[0] - v0[0], average[1] - v0[1],
                                 average[2] - v0[2]};

    const REAL Etest_dot_normal =
        KERNEL_DOT_PRODUCT_3D(Etest[0], Etest[1], Etest[2], normal_vector[0],
                              normal_vector[1], normal_vector[2]);

    if (Etest_dot_normal > 0.0) {
      normal_vector[0] *= -1.0;
      normal_vector[1] *= -1.0;
      normal_vector[2] *= -1.0;
    }
  }
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

void DMPlexHelper::get_global_face_index_bounds(INT &bound_lower,
                                                INT &bound_upper) {

  bound_lower = this->boundary_index_bound_lower;
  bound_upper = this->boundary_index_bound_upper;
}

bool DMPlexHelper::normal_points_towards_point(const PetscInt p0,
                                               const PetscInt p1,
                                               const PetscInt p2,
                                               const PetscInt point) {

  auto get_coords = [](DM dm, PetscInt petsc_index) {
    const PetscScalar *tmp;
    PetscScalar *vertices = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;

    PETSCCHK(DMPlexGetCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                      &tmp, &vertices));
    const int num_vertices = num_coords / 3;
    NESOASSERT(num_vertices == 1, "Expected a point.");
    std::vector<PetscScalar> h_vertices;
    h_vertices.reserve(num_coords);
    for (PetscInt ix = 0; ix < num_coords; ix++) {
      h_vertices.push_back(vertices[ix]);
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, petsc_index, &is_dg, &num_coords,
                                          &tmp, &vertices));

    return h_vertices;
  };

  auto v0 = get_coords(dm, p0);
  auto v1 = get_coords(dm, p1);
  auto v2 = get_coords(dm, p2);
  auto d = get_coords(dm, point);

  std::vector<REAL> n01(3);
  std::vector<REAL> n02(3);
  std::vector<REAL> vd(3);
  for (int dx = 0; dx < 3; dx++) {
    n01.at(dx) = v1.at(dx) - v0.at(dx);
    n02.at(dx) = v2.at(dx) - v0.at(dx);
    vd.at(dx) = d.at(dx) - v0.at(dx);
  }
  std::vector<REAL> n = {0.0, 0.0, 0.0};

  KERNEL_CROSS_PRODUCT_3D(n01[0], n01[1], n01[2], n02[0], n02[1], n02[2], n[0],
                          n[1], n[2]);
  const REAL dd = KERNEL_DOT_PRODUCT_3D(n[0], n[1], n[2], vd[0], vd[1], vd[2]);

  return dd >= 0.0;
}

void DMPlexHelper::get_vertex_neighbours(const PetscInt point_index,
                                         std::vector<PetscInt> &neighbours) {

  neighbours.clear();
  PetscInt support_size = 0;
  PETSCCHK(DMPlexGetSupportSize(dm, point_index, &support_size));
  const PetscInt *support = nullptr;
  PETSCCHK(DMPlexGetSupport(dm, point_index, &support));

  for (PetscInt sx = 0; sx < support_size; sx++) {
    PetscInt cone_size = 0;
    const PetscInt support_point = support[sx];
    PETSCCHK(DMPlexGetConeSize(dm, support_point, &cone_size));
    NESOASSERT(cone_size == 2, "Expected support point to be an edge.");
    const PetscInt *support_cone = nullptr;
    PETSCCHK(DMPlexGetCone(dm, support_point, &support_cone));
    const PetscInt p0 = support_cone[0];
    const PetscInt p1 = support_cone[1];

    if (p0 == point_index) {
      neighbours.push_back(p1);
    } else {
      neighbours.push_back(p0);
    }
  }
}

void DMPlexHelper::get_canonical_vertex_order(const PetscInt point,
                                              std::vector<PetscInt> &order) {

  order.clear();

  bool remake = this->map_point_to_vertex_order.count(point) == 0;
  if (this->map_point_to_vertex_type.count(point) == 0) {
    remake = true;
  }
  if ((this->map_point_to_vertex_type.count(point)) &&
      (this->map_point_to_vertex_type.at(point) !=
       this->get_point_type(point))) {
    remake = true;
  }

  if (remake) {

    PetscInt depth = -1;
    PETSCCHK(DMPlexGetPointDepth(dm, point, &depth));
    const PetscInt *cone = nullptr;
    PetscInt cone_size = 0;
    PETSCCHK(DMPlexGetConeSize(dm, point, &cone_size));
    if (cone_size > 0) {
      PETSCCHK(DMPlexGetCone(dm, point, &cone));
    }

    auto point_type = get_point_type(point);

    std::vector<std::vector<PetscInt>> faces;
    std::set<PetscInt> vertex_points;
    for (PetscInt fx = 0; fx < cone_size; fx++) {
      std::vector<PetscInt> t;
      this->get_canonical_vertex_order(cone[fx], t);
      for (auto tx : t) {
        vertex_points.insert(tx);
      }
      faces.push_back(t);
    }

    if (point_type == DM_POLYTOPE_POINT) {
      order.push_back(point);
    } else if (point_type == DM_POLYTOPE_SEGMENT) {
      order.push_back(cone[0]);
      order.push_back(cone[1]);
    } else if (point_type == DM_POLYTOPE_POINT_PRISM_TENSOR) {
      order.push_back(cone[0]);
      order.push_back(cone[1]);
    } else if ((point_type == DM_POLYTOPE_TRIANGLE) ||
               (point_type == DM_POLYTOPE_QUADRILATERAL) ||
               (point_type == DM_POLYTOPE_SEG_PRISM_TENSOR)) {

      std::map<PetscInt, std::set<PetscInt>> map_vertex_to_neighbours;

      PetscInt first_vertex = -1;
      for (PetscInt edgex = 0; edgex < cone_size; edgex++) {
        const PetscInt edge = cone[edgex];
        std::vector<PetscInt> edge_cone;
        this->get_canonical_vertex_order(edge, edge_cone);

        const PetscInt v0 = edge_cone.at(0);
        const PetscInt v1 = edge_cone.at(1);
        if (edgex == 0) {
          first_vertex = v0;
        }

        map_vertex_to_neighbours[v0].insert(v1);
        map_vertex_to_neighbours[v1].insert(v0);
      }

      PetscInt current_vertex = first_vertex;
      for (int edgex = 0; edgex < cone_size; edgex++) {

        order.push_back(current_vertex);
        // get a neighbour vertex
        const PetscInt next_vertex =
            *map_vertex_to_neighbours.at(current_vertex).begin();
        // Remove the current point from the neighbours of the next point such
        // that the loop never travels backwards.
        map_vertex_to_neighbours.at(next_vertex).erase(current_vertex);

        current_vertex = next_vertex;
      }

      if (point_type == DM_POLYTOPE_SEG_PRISM_TENSOR) {
        const PetscInt t2 = order.at(2);
        const PetscInt t3 = order.at(3);
        order.at(2) = t3;
        order.at(3) = t2;
      }

    } else if (point_type == DM_POLYTOPE_TETRAHEDRON) {

      auto bottom_face = faces.at(0);
      for (auto tx : bottom_face) {
        vertex_points.erase(tx);
      }
      NESOASSERT(vertex_points.size() == 1, "Expected one remaining point.");
      const PetscInt point3 = *vertex_points.begin();

      const bool correct_order = normal_points_towards_point(
          bottom_face.at(0), bottom_face.at(2), bottom_face.at(1), point3);

      if (correct_order) {
        order.push_back(bottom_face.at(0));
        order.push_back(bottom_face.at(1));
        order.push_back(bottom_face.at(2));
        order.push_back(point3);
      } else {
        order.push_back(bottom_face.at(2));
        order.push_back(bottom_face.at(1));
        order.push_back(bottom_face.at(0));
        order.push_back(point3);
      }
    } else if (point_type == DM_POLYTOPE_PYRAMID) {
      // There is one quad and this is the base.

      std::vector<PetscInt> bottom_face;
      for (auto &fx : faces) {
        if (fx.size() == 4) {
          bottom_face = fx;
        }
      }
      NESOASSERT(bottom_face.size() == 4, "Failed to find Pyramid base.");
      for (auto tx : bottom_face) {
        vertex_points.erase(tx);
      }
      NESOASSERT(vertex_points.size() == 1, "Expected one remaining point.");
      const PetscInt point4 = *vertex_points.begin();

      const bool correct_order = normal_points_towards_point(
          bottom_face.at(0), bottom_face.at(3), bottom_face.at(1), point4);

      if (correct_order) {
        order.push_back(bottom_face.at(0));
        order.push_back(bottom_face.at(1));
        order.push_back(bottom_face.at(2));
        order.push_back(bottom_face.at(3));
        order.push_back(point4);
      } else {
        order.push_back(bottom_face.at(3));
        order.push_back(bottom_face.at(2));
        order.push_back(bottom_face.at(1));
        order.push_back(bottom_face.at(0));
        order.push_back(point4);
      }
    } else if ((point_type == DM_POLYTOPE_TRI_PRISM) ||
               (point_type == DM_POLYTOPE_TRI_PRISM_TENSOR)) {

      std::vector<PetscInt> top_face;
      std::vector<PetscInt> bottom_face;
      for (auto &fx : faces) {
        // Only consider the triangles.
        if (fx.size() == 3) {
          if (top_face.size() == 0) {
            top_face = fx;
          } else if (bottom_face.size() == 0) {
            bottom_face = fx;
          }
        } else {
          NESOASSERT(fx.size() == 4, "Remaining faces should be quads.");
        }
      }
      NESOASSERT(top_face.size() == 3, "Failed to find top face.");
      NESOASSERT(bottom_face.size() == 3, "Failed to find bottom face.");

      const bool bottom_points_inwards =
          normal_points_towards_point(bottom_face.at(0), bottom_face.at(2),
                                      bottom_face.at(1), top_face.at(0));

      // tri prism bottom face is clockwise for prism and anticlockwise for
      // tensor prism.
      if ((!bottom_points_inwards) && (point_type == DM_POLYTOPE_TRI_PRISM)) {
        std::reverse(bottom_face.begin(), bottom_face.end());
      }
      if ((bottom_points_inwards) &&
          (point_type == DM_POLYTOPE_TRI_PRISM_TENSOR)) {
        std::reverse(bottom_face.begin(), bottom_face.end());
      }

      const bool top_normal_upwards = !normal_points_towards_point(
          top_face.at(0), top_face.at(1), top_face.at(2), bottom_face.at(0));
      // tri prism and the tensor version have the same top face ordering.
      if ((!top_normal_upwards)) {
        std::reverse(top_face.begin(), top_face.end());
      }

      const PetscInt p0 = bottom_face.at(0);
      NESOASSERT(this->get_point_type(p0) == DM_POLYTOPE_POINT,
                 "Expected p0 to be a point.");
      std::vector<PetscInt> neighbours;
      get_vertex_neighbours(p0, neighbours);

      PetscInt p3;
      for (auto &nx : neighbours) {
        if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
          p3 = nx;
          break;
        }
      }

      auto iterator_start_top = std::find(top_face.begin(), top_face.end(), p3);
      const std::size_t index_start_top = iterator_start_top - top_face.begin();
      NESOASSERT(index_start_top < 3, "Failed to find starting top index");

      const PetscInt p4 = top_face.at((index_start_top + 1) % 3);
      const PetscInt p5 = top_face.at((index_start_top + 2) % 3);

      PetscInt to_test;
      get_vertex_neighbours(bottom_face.at(1), neighbours);
      for (auto nx : neighbours) {
        if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
          to_test = nx;
          break;
        }
      }

      const PetscInt correct = (point_type == DM_POLYTOPE_TRI_PRISM) ? p5 : p4;
      NESOASSERT(to_test == correct,
                 "Failed to find consistent loop for top and bottom faces.");

      order.push_back(bottom_face.at(0));
      order.push_back(bottom_face.at(1));
      order.push_back(bottom_face.at(2));
      order.push_back(p3);
      order.push_back(p4);
      order.push_back(p5);
    } else if ((point_type == DM_POLYTOPE_HEXAHEDRON) ||
               (point_type == DM_POLYTOPE_QUAD_PRISM_TENSOR)) {

      auto bottom_face = faces.at(0);
      std::set<PetscInt> bottom_face_set;
      for (auto &fx : bottom_face) {
        bottom_face_set.insert(fx);
      }

      std::vector<PetscInt> top_face;

      for (auto &fx : faces) {
        NESOASSERT(fx.size() == 4, "Expected all faces to be quads.");

        bool top_face_candidate = true;
        for (auto px : fx) {
          if (bottom_face_set.count(px)) {
            top_face_candidate = false;
            break;
          }
        }
        if (top_face_candidate) {
          top_face = fx;
          break;
        }
      }

      NESOASSERT(top_face.size() == 4, "Failed to find a top face.");
      for (auto &px : top_face) {
        NESOASSERT(bottom_face_set.count(px) == 0,
                   "Top face candidate has a point from the bottom face.");
      }

      const bool bottom_points_inwards =
          normal_points_towards_point(bottom_face.at(0), bottom_face.at(1),
                                      bottom_face.at(3), top_face.at(0));

      if (bottom_points_inwards && (point_type == DM_POLYTOPE_HEXAHEDRON)) {
        std::reverse(bottom_face.begin(), bottom_face.end());
      }
      if (!bottom_points_inwards &&
          (point_type == DM_POLYTOPE_QUAD_PRISM_TENSOR)) {
        std::reverse(bottom_face.begin(), bottom_face.end());
      }

      const bool top_points_inwards = normal_points_towards_point(
          top_face.at(0), top_face.at(1), top_face.at(3), bottom_face.at(0));

      if (top_points_inwards) {
        std::reverse(top_face.begin(), top_face.end());
      }

      if (point_type == DM_POLYTOPE_HEXAHEDRON) {
        const bool bottom0 =
            normal_points_towards_point(bottom_face.at(0), bottom_face.at(3),
                                        bottom_face.at(1), top_face.at(0));
        NESOASSERT(bottom0, "Bottom normal check failed.");
      }

      if (point_type == DM_POLYTOPE_QUAD_PRISM_TENSOR) {
        const bool bottom0 =
            normal_points_towards_point(bottom_face.at(0), bottom_face.at(1),
                                        bottom_face.at(3), top_face.at(0));
        NESOASSERT(bottom0, "Bottom normal check failed.");
      }

      const bool top0 = !normal_points_towards_point(
          top_face.at(0), top_face.at(1), top_face.at(3), bottom_face.at(0));

      NESOASSERT(top0, "Top normal check failed.");

      const PetscInt p0 = bottom_face.at(0);
      std::vector<PetscInt> neighbours;
      get_vertex_neighbours(p0, neighbours);
      PetscInt p4;
      bool p4_found = false;
      for (auto &nx : neighbours) {
        if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
          NESOASSERT(!p4_found, "p4 was already found.");
          p4_found = true;
          p4 = nx;
        }
      }

      auto iterator_start_top = std::find(top_face.begin(), top_face.end(), p4);
      const std::size_t index_start_top = iterator_start_top - top_face.begin();
      NESOASSERT(index_start_top < 4, "Failed to find starting top index");
      NESOASSERT(top_face.at(index_start_top) == p4,
                 "p4 consistency check failed.");
      const PetscInt p5 = top_face.at((index_start_top + 1) % 4);
      const PetscInt p6 = top_face.at((index_start_top + 2) % 4);
      const PetscInt p7 = top_face.at((index_start_top + 3) % 4);

      PetscInt to_test;
      get_vertex_neighbours(bottom_face.at(1), neighbours);
      for (auto nx : neighbours) {
        if (std::find(top_face.begin(), top_face.end(), nx) != top_face.end()) {
          to_test = nx;
          break;
        }
      }

      const PetscInt correct = (point_type == DM_POLYTOPE_HEXAHEDRON) ? p7 : p5;
      NESOASSERT(to_test == correct,
                 "Failed to find consistent loop for top and bottom faces.");

      order.push_back(bottom_face.at(0));
      order.push_back(bottom_face.at(1));
      order.push_back(bottom_face.at(2));
      order.push_back(bottom_face.at(3));
      order.push_back(p4);
      order.push_back(p5);
      order.push_back(p6);
      order.push_back(p7);

    } else {
      NESOASSERT(false, "Unknown point type.");
    }

    this->map_point_to_vertex_order[point] = order;

  } else {
    order.insert(order.end(), this->map_point_to_vertex_order.at(point).begin(),
                 this->map_point_to_vertex_order.at(point).end());
  }
}

} // namespace NESO::Particles::PetscInterface

#endif
