#include <neso_particles/cartesian_mesh/cartesian_h_mesh.hpp>

namespace NESO::Particles {

CartesianHMesh::CartesianHMesh(MPI_Comm comm, const int ndim,
                               std::vector<int> &dims, const double extent,
                               const int subdivision_order,
                               const int stencil_width)
    : stencil_width(stencil_width), ndim(ndim), dims(dims),
      subdivision_order(subdivision_order), cell_width_coarse(extent),
      cell_width_fine(extent / ((double)std::pow(2, subdivision_order))),
      inverse_cell_width_coarse(1.0 / extent),
      inverse_cell_width_fine(((double)std::pow(2, subdivision_order)) /
                              extent),
      ncells_coarse(reduce_mul(ndim, dims)),
      ncells_fine(std::pow(std::pow(2, subdivision_order), ndim)),
      single_cell_mode(false) {

  // basic error checking of inputs
  NESOASSERT(ndim > 0, "ndim less than 1");
  NESOASSERT(dims.size() >= static_cast<std::size_t>(ndim),
             "vector of dims too small");
  for (int dimx = 0; dimx < ndim; dimx++) {
    NESOASSERT(dims[dimx] > 0, "Dim size is <= 0 in a direction.");
  }
  NESOASSERT(cell_width_coarse > 0.0, "Extent <= 0.0 passed");
  NESOASSERT(subdivision_order >= 0, "Negative subdivision order passed.");

  // mpi decompose
  // mpi_dims has monotonically decreasing order
  int rank, size;
  MPICHK(MPI_Comm_size(comm, &size));
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Dims_create(size, ndim, mpi_dims));

  for (int dimx = 0; dimx < ndim; dimx++) {
    cell_counts[dimx] = dims[dimx] * std::pow(2, subdivision_order);
    global_extents[dimx] = extent * dims[dimx];
  }
  // direction with most cells first to match mpi_dims order
  auto cell_count_ordering = reverse_argsort(cell_counts);

  // reorder the mpi_dims to match the actual domain
  std::vector<int> mpi_dims_reordered(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    mpi_dims_reordered[cell_count_ordering[dimx]] = mpi_dims[dimx];
  }

  // create MPI cart comm with a decomposition that roughly makes sense for
  // the shape of the domain
  MPICHK(MPI_Cart_create(comm, ndim, mpi_dims_reordered.data(), periods, 1,
                         &comm_cart));
  this->allocated = true;

  // get the information about the cart comm that was actually created
  MPICHK(MPI_Cart_get(comm_cart, ndim, mpi_dims, periods, coords));

  // in each dimension compute the portion owned by this rank
  for (int dimx = 0; dimx < ndim; dimx++) {
    get_decomp_1d(mpi_dims[dimx], cell_counts[dimx], coords[dimx],
                  &cell_starts[dimx], &cell_ends[dimx]);
  }

  std::vector<double> origin(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    origin[dimx] = 0.0;
  }
  // for this mesh the hierarchy is simply a copy of the mesh
  this->mesh_hierarchy = std::make_shared<MeshHierarchy>(
      comm_cart, ndim, dims, origin, cell_width_coarse, subdivision_order);

  // setup the hierarchy

  // mesh tuple index
  INT index_mesh[3];
  // mesh_hierarchy tuple index
  INT index_mh[6];

  mesh_hierarchy->claim_initialise();

  // store the size of the local subdomain such that local linear indices can
  // be computed
  for (int dimx = 0; dimx < ndim; dimx++) {
    const int tmp_width = cell_ends[dimx] - cell_starts[dimx];
    NESOASSERT(tmp_width > 0,
               "A domain of width <= 0 cells does not make sense");
    this->cell_counts_local[dimx] = tmp_width;
  }

  // loop over owned cells
  this->cell_count = 0;
  for (int cz = cell_starts[2]; cz < cell_ends[2]; cz++) {
    index_mesh[2] = cz;
    for (int cy = cell_starts[1]; cy < cell_ends[1]; cy++) {
      index_mesh[1] = cy;
      for (int cx = cell_starts[0]; cx < cell_ends[0]; cx++) {
        index_mesh[0] = cx;
        // convert mesh tuple index to mesh hierarchy tuple index
        mesh_tuple_to_mh_tuple(index_mesh, index_mh);
        // convert mesh hierarchy tuple index to global linear index in the
        // MeshHierarchy
        const INT index_global =
            mesh_hierarchy->tuple_to_linear_global(index_mh);
        // claim ownership of the current cell in the MeshHierarchy
        mesh_hierarchy->claim_cell(index_global, 1);
        this->cell_count++;
      }
    }
  }

  mesh_hierarchy->claim_finalise();

  // get the MPI ranks owning cells within the stencil region

  int stencil_cell_starts[3] = {0, 0, 0};
  int stencil_cell_ends[3] = {1, 1, 1};

  for (int dimx = 0; dimx < this->ndim; dimx++) {
    stencil_cell_starts[dimx] = this->cell_starts[dimx] - this->stencil_width;
    stencil_cell_ends[dimx] = this->cell_ends[dimx] + this->stencil_width;
  }

  std::set<int> neighbour_ranks_set;
  for (int scz = stencil_cell_starts[2]; scz < stencil_cell_ends[2]; scz++) {
    const int cz = (scz + this->stencil_width * this->cell_counts[2]) %
                   MAX(this->cell_counts[2], 1);
    index_mesh[2] = cz;
    for (int scy = stencil_cell_starts[1]; scy < stencil_cell_ends[1]; scy++) {
      const int cy = (scy + this->stencil_width * this->cell_counts[1]) %
                     MAX(this->cell_counts[1], 1);
      index_mesh[1] = cy;
      for (int scx = stencil_cell_starts[0]; scx < stencil_cell_ends[0];
           scx++) {
        const int cx = (scx + this->stencil_width * this->cell_counts[0]) %
                       this->cell_counts[0];
        index_mesh[0] = cx;
        // convert mesh tuple index to mesh hierarchy tuple index
        mesh_tuple_to_mh_tuple(index_mesh, index_mh);
        // convert mesh hierarchy tuple index to global linear index in the
        // MeshHierarchy
        const INT index_global =
            mesh_hierarchy->tuple_to_linear_global(index_mh);
        // claim ownership of the current cell in the MeshHierarchy
        const int owning_rank = this->mesh_hierarchy->get_owner(index_global);
        neighbour_ranks_set.insert(owning_rank);
      }
    }
  }

  this->neighbour_ranks.reserve(neighbour_ranks_set.size());
  for (auto rankx : neighbour_ranks_set) {
    this->neighbour_ranks.push_back(rankx);
  }

  // set number of geoms per face
  const auto ncells_dim_fine = this->mesh_hierarchy->ncells_dim_fine;
  if (this->ndim == 2) {
    this->num_geoms_per_face[0] = dims[0] * ncells_dim_fine;
    this->num_geoms_per_face[1] = dims[1] * ncells_dim_fine;
    this->num_geoms_per_face[2] = dims[0] * ncells_dim_fine;
    this->num_geoms_per_face[3] = dims[1] * ncells_dim_fine;

    const INT total_boundary_cells = std::accumulate(
        this->num_geoms_per_face.begin(), this->num_geoms_per_face.end(), 0);
    NESOASSERT(total_boundary_cells ==
                   ncells_dim_fine * (dims[0] + dims[1] + dims[0] + dims[1]),
               "Incorrect number of boundary cells.");
    this->num_face_geoms = static_cast<int>(total_boundary_cells);

    this->face_strides0[0] = dims[0] * ncells_dim_fine;
    this->face_strides0[1] = dims[1] * ncells_dim_fine;
    this->face_strides0[2] = dims[0] * ncells_dim_fine;
    this->face_strides0[3] = dims[1] * ncells_dim_fine;
    this->face_strides1[0] = 1;
    this->face_strides1[1] = 1;
    this->face_strides1[2] = 1;
    this->face_strides1[3] = 1;

  } else if (this->ndim == 3) {
    this->num_geoms_per_face[0] =
        dims[0] * dims[2] * std::pow(ncells_dim_fine, 2);
    this->num_geoms_per_face[1] =
        dims[1] * dims[2] * std::pow(ncells_dim_fine, 2);
    this->num_geoms_per_face[2] =
        dims[0] * dims[2] * std::pow(ncells_dim_fine, 2);
    this->num_geoms_per_face[3] =
        dims[1] * dims[2] * std::pow(ncells_dim_fine, 2);
    this->num_geoms_per_face[4] =
        dims[0] * dims[1] * std::pow(ncells_dim_fine, 2);
    this->num_geoms_per_face[5] =
        dims[0] * dims[1] * std::pow(ncells_dim_fine, 2);

    const INT total_boundary_cells = std::accumulate(
        this->num_geoms_per_face.begin(), this->num_geoms_per_face.end(), 0);
    NESOASSERT(total_boundary_cells ==
                   ncells_dim_fine * ncells_dim_fine *
                       (dims[0] * dims[2] + dims[1] * dims[2] +
                        dims[0] * dims[2] + dims[1] * dims[2] +
                        dims[0] * dims[1] + dims[0] * dims[1]),
               "Incorrect number of boundary cells.");

    this->num_face_geoms = static_cast<int>(total_boundary_cells);

    this->face_strides0[0] = dims[0] * ncells_dim_fine;
    this->face_strides0[1] = dims[1] * ncells_dim_fine;
    this->face_strides0[2] = dims[0] * ncells_dim_fine;
    this->face_strides0[3] = dims[1] * ncells_dim_fine;
    this->face_strides0[4] = dims[0] * ncells_dim_fine;
    this->face_strides0[5] = dims[0] * ncells_dim_fine;

    this->face_strides1[0] = dims[2] * ncells_dim_fine;
    this->face_strides1[1] = dims[2] * ncells_dim_fine;
    this->face_strides1[2] = dims[2] * ncells_dim_fine;
    this->face_strides1[3] = dims[2] * ncells_dim_fine;
    this->face_strides1[4] = dims[1] * ncells_dim_fine;
    this->face_strides1[5] = dims[1] * ncells_dim_fine;
  }

  int total = 0;
  for (int ix = 0; ix < 6; ix++) {
    num_geoms_per_face_exscan[ix] = total;
    total += this->num_geoms_per_face[ix];
    num_geoms_per_face_incscan[ix] = total;
  }

  this->compute_owned_face_indices();
  MPICHK(MPI_Comm_split(this->comm_cart,
                        this->owned_face_indices.size() ? 0 : MPI_UNDEFINED, 0,
                        &this->comm_faces));
}

void CartesianHMesh::compute_owned_face_indices() {

  NESOASSERT((1 < this->ndim) && (this->ndim < 4),
             "Unexpected number of dimensions.");

  // This is not amazing but is at least has a cost that is proportional to the
  // size of the locally owned domain instead of scaling with the global size.

  const int l0_starts[6] = {this->cell_starts[0], this->cell_starts[1],
                            this->cell_starts[0], this->cell_starts[1],
                            this->cell_starts[0], this->cell_starts[0]};

  const int l0_ends[6] = {this->cell_ends[0], this->cell_ends[1],
                          this->cell_ends[0], this->cell_ends[1],
                          this->cell_ends[0], this->cell_ends[0]};

  const int l1_starts[6] = {this->cell_starts[2], this->cell_starts[2],
                            this->cell_starts[2], this->cell_starts[2],
                            this->cell_starts[1], this->cell_starts[1]};

  const int l1_ends[6] = {this->cell_ends[2], this->cell_ends[2],
                          this->cell_ends[2], this->cell_ends[2],
                          this->cell_ends[1], this->cell_ends[1]};

  const int num_faces = this->ndim == 2 ? 4 : 6;
  std::set<INT> owned_face_indices_set;

  int rank;
  MPICHK(MPI_Comm_rank(this->comm_cart, &rank));

  for (INT fx = 0; fx < num_faces; fx++) {
    for (INT l1 = l1_starts[fx]; l1 < l1_ends[fx]; l1++) {
      for (INT l0 = l0_starts[fx]; l0 < l0_ends[fx]; l0++) {
        INT index_tuple[3] = {fx, l0, l1};
        const INT linear_face_index =
            this->get_face_linear_index_from_tuple(index_tuple);
        const int owning_rank =
            this->get_face_id_owning_rank(linear_face_index);
        if (owning_rank == rank) {
          owned_face_indices_set.insert(linear_face_index);
        }
      }
    }
  }

  this->owned_face_indices.reserve(owned_face_indices_set.size());
  for (auto &ix : owned_face_indices_set) {
    this->owned_face_indices.push_back(ix);
  }
}

MPI_Comm CartesianHMesh::get_comm() { return this->comm_cart; }
int CartesianHMesh::get_ndim() { return this->ndim; };
std::vector<int> &CartesianHMesh::get_dims() { return this->dims; }
int CartesianHMesh::get_subdivision_order() { return this->subdivision_order; }
double CartesianHMesh::get_cell_width_coarse() {
  return this->cell_width_coarse;
}
double CartesianHMesh::get_cell_width_fine() { return this->cell_width_fine; }
double CartesianHMesh::get_inverse_cell_width_coarse() {
  return this->inverse_cell_width_coarse;
}
double CartesianHMesh::get_inverse_cell_width_fine() {
  return this->inverse_cell_width_fine;
}
int CartesianHMesh::get_ncells_coarse() { return this->ncells_coarse; }
int CartesianHMesh::get_ncells_fine() { return this->ncells_fine; }

int CartesianHMesh::get_cell_count() {
  if (this->single_cell_mode) {
    return 1;
  } else {
    return this->cell_count;
  }
}

int CartesianHMesh::get_cart_cell_count() { return this->cell_count; }

std::shared_ptr<MeshHierarchy> CartesianHMesh::get_mesh_hierarchy() {
  return this->mesh_hierarchy;
}

void CartesianHMesh::mesh_tuple_to_mh_tuple(const INT *index_mesh,
                                            INT *index_mh) {
  for (int dimx = 0; dimx < ndim; dimx++) {
    auto pq = std::div((long long)index_mesh[dimx],
                       (long long)mesh_hierarchy->ncells_dim_fine);
    index_mh[dimx] = pq.quot;
    index_mh[dimx + ndim] = pq.rem;
  }
}

void CartesianHMesh::free() {
  int flag;
  MPICHK(MPI_Initialized(&flag))
  if (allocated && flag) {
    if (this->comm_faces != MPI_COMM_NULL) {
      MPICHK(MPI_Comm_free(&this->comm_faces));
    }

    MPICHK(MPI_Comm_free(&this->comm_cart))
    this->comm_cart = MPI_COMM_NULL;
  }
  this->allocated = false;
  mesh_hierarchy->free();
}

std::vector<int> &CartesianHMesh::get_local_communication_neighbours() {
  return this->neighbour_ranks;
}

void CartesianHMesh::get_point_in_subdomain(double *point) {
  for (int dimx = 0; dimx < ndim; dimx++) {
    const double start = this->cell_starts[dimx] * this->cell_width_fine;
    const double end = this->cell_ends[dimx] * this->cell_width_fine;
    point[dimx] = 0.5 * (start + end);
  }
}

std::vector<std::array<int, 3>> CartesianHMesh::get_owned_cells() {
  std::vector<std::array<int, 3>> cells(this->cell_count);
  int index = 0;
  for (int cz = cell_starts[2]; cz < cell_ends[2]; cz++) {
    for (int cy = cell_starts[1]; cy < cell_ends[1]; cy++) {
      for (int cx = cell_starts[0]; cx < cell_ends[0]; cx++) {
        cells[index][0] = cx;
        cells[index][1] = cy;
        cells[index][2] = cz;
        index++;
      }
    }
  }
  return cells;
}

std::array<int, 3>
CartesianHMesh::get_global_cell_tuple_index(const INT linear_cell_index) {

  NESOASSERT((0 <= linear_cell_index) && (linear_cell_index < this->cell_count),
             "Bad linear cell index.");

  std::array<int, 3> index = {0, 0, 0};
  int w[3] = {1, 1, 1};

  for (int dx = 0; dx < this->ndim; dx++) {
    w[dx] = this->cell_ends[dx] - this->cell_starts[dx];
  }

  INT linear_cell_index_t = linear_cell_index;
  for (int dx = 0; dx < this->ndim; dx++) {
    INT cx = linear_cell_index_t % w[dx];
    linear_cell_index_t -= cx;
    linear_cell_index_t /= w[dx];
    index[dx] = this->cell_starts[dx] + static_cast<int>(cx);
  }

  return index;
}

int CartesianHMesh::get_mesh_tuple_owning_rank(const INT *index_mesh) {

  std::vector<int> coords_tmp(this->ndim);

  for (int dx = 0; dx < this->ndim; dx++) {
    NESOASSERT((0 <= index_mesh[dx]) &&
                   (index_mesh[dx] <
                    this->dims[dx] * this->mesh_hierarchy->ncells_dim_fine),
               "Bad index passed.");

    coords_tmp[dx] = static_cast<int>(
        get_decomp_1d_inverse(static_cast<std::size_t>(this->mpi_dims[dx]),
                              static_cast<std::size_t>(this->cell_counts[dx]),
                              static_cast<std::size_t>(index_mesh[dx])));
  }

  int owning_rank = -1;

  MPICHK(MPI_Cart_rank(this->comm_cart, coords_tmp.data(), &owning_rank));
  return owning_rank;
}

void CartesianHMesh::get_face_id_as_tuple(const INT face_id,
                                          INT *face_index_tuple) {

  const int num_faces = this->ndim == 2 ? 4 : 6;

  const int num_face_indices = std::accumulate(
      num_geoms_per_face.begin(), num_geoms_per_face.begin() + num_faces, 0);

  NESOASSERT((0 <= face_id) && (face_id < num_face_indices),
             "Bad linear face id passed.");

  int face_containing = -1;
  for (int facex = 0; facex < num_faces; facex++) {
    if (face_id < this->num_geoms_per_face_incscan[facex]) {
      face_containing = facex;
      break;
    }
  }

  NESOASSERT(face_containing >= 0, "Failed to find face (lower bound).");
  NESOASSERT(face_containing < num_faces, "Failed to find face (upper bound).");

  // id local to each face
  int local_face_id =
      face_id - this->num_geoms_per_face_exscan[face_containing];

  // local face ids are lexicographically indexed
  const int l0 = local_face_id % this->face_strides0[face_containing];
  const int l1 = local_face_id / this->face_strides0[face_containing];

  NESOASSERT((0 <= l1) && (l1 < this->face_strides1[face_containing]),
             "Bad l1 computed.");

  NESOASSERT((l0 + l1 * this->face_strides0[face_containing]) == local_face_id,
             "Failed to compute geom index correctly (local id)");

  NESOASSERT(this->num_geoms_per_face_exscan[face_containing] + local_face_id ==
                 face_id,
             "Failed to compute geom index correctly (local id + offset)");

  face_index_tuple[0] = face_containing;
  face_index_tuple[1] = l0;
  face_index_tuple[2] = l1;
}

void CartesianHMesh::get_mesh_tuple_owning_face_tuple(
    const INT *face_index_tuple, INT *mesh_tuple) {
  const INT face_index = face_index_tuple[0];
  const INT l0 = face_index_tuple[1];

  if (this->ndim == 2) {
    const int index_lhs0[4] = {0, 1, 0, 1};

    const int index_lhs1[4] = {1, 0, 1, 0};
    const INT last_values[4] = {0, this->cell_counts[0] - 1,
                                this->cell_counts[1] - 1, 0};
    mesh_tuple[index_lhs0[face_index]] = l0;
    mesh_tuple[index_lhs1[face_index]] = last_values[face_index];
  } else {
    const INT l1 = face_index_tuple[2];

    const int index_lhs0[6] = {0, 1, 0, 1, 0, 0};
    const int index_lhs1[6] = {2, 2, 2, 2, 1, 1};
    const int index_lhs2[6] = {1, 0, 1, 0, 2, 2};
    const INT last_values[6] = {
        0, this->cell_counts[0] - 1, this->cell_counts[1] - 1, 0,
        0, this->cell_counts[2] - 1};

    mesh_tuple[index_lhs0[face_index]] = l0;
    mesh_tuple[index_lhs1[face_index]] = l1;
    mesh_tuple[index_lhs2[face_index]] = last_values[face_index];
  }
}

INT CartesianHMesh::get_face_linear_index_from_tuple(
    const INT *face_index_tuple) {

  const INT offset = std::accumulate(
      this->num_geoms_per_face.begin(),
      this->num_geoms_per_face.begin() + face_index_tuple[0], 0);

  if (this->ndim == 2) {
    return offset + face_index_tuple[1];
  } else {
    return offset + face_index_tuple[1] +
           face_index_tuple[2] * this->face_strides0[face_index_tuple[0]];
  }
}

int CartesianHMesh::get_face_id_owning_rank(const INT face_id) {
  NESOASSERT(this->ndim == 2 || this->ndim == 3,
             "Unexpected number of dimensions.");
  NESOASSERT((0 <= face_id) && (face_id < this->num_face_geoms), "Bad face id");
  if (!this->cache_map_face_id_to_rank.count(face_id)) {

    INT face_tuple[3] = {0, 0, 0};
    INT mesh_tuple[3] = {0, 0, 0};

    this->get_face_id_as_tuple(face_id, face_tuple);
    this->get_mesh_tuple_owning_face_tuple(face_tuple, mesh_tuple);
    const int owning_rank = this->get_mesh_tuple_owning_rank(mesh_tuple);

    this->cache_map_face_id_to_rank[face_id] = owning_rank;
  }
  return this->cache_map_face_id_to_rank[face_id];
}

std::vector<double>
CartesianHMesh::get_vtk_cell_points(const INT *index_tuple) {

  std::vector<std::array<int, 3>> offsets;
  offsets.reserve(1 << this->ndim);

  // The ordering matters here for VTK.
  offsets.push_back({0, 0, 0});
  offsets.push_back({1, 0, 0});
  if (this->ndim > 1) {
    offsets.push_back({1, 1, 0});
    offsets.push_back({0, 1, 0});
  }
  if (this->ndim > 2) {
    offsets.push_back({0, 0, 1});
    offsets.push_back({1, 0, 1});
    offsets.push_back({1, 1, 1});
    offsets.push_back({0, 1, 1});
  }

  std::vector<double> points;
  points.reserve(3 * (1 << this->ndim));

  const auto width = this->cell_width_fine;

  for (auto &ox : offsets) {
    // The VTK interface is in R^3.
    for (int dx = 0; dx < 3; dx++) {
      points.push_back(index_tuple[dx] * width + ox[dx] * width);
    }
  }

  return points;
}

std::vector<double> CartesianHMesh::get_vtk_cell_points(const INT index) {

  auto cell_tuple = this->get_global_cell_tuple_index(index);
  INT index_tuple[3] = {cell_tuple[0], cell_tuple[1], cell_tuple[2]};

  return this->get_vtk_cell_points(index_tuple);
}

std::vector<VTK::UnstructuredCell> CartesianHMesh::get_vtk_cell_data() {
  std::vector<VTK::UnstructuredCell> vtk_cell_data(this->cell_count);
  for (int cx = 0; cx < this->cell_count; cx++) {
    vtk_cell_data.at(cx).num_points = this->ndim == 2 ? 4 : 8;
    vtk_cell_data.at(cx).points = this->get_vtk_cell_points(cx);
    vtk_cell_data.at(cx).cell_type =
        this->ndim == 2 ? VTK::CellType::quadrilateral : VTK::CellType::hex;
  }

  return vtk_cell_data;
}

std::vector<double>
CartesianHMesh::get_vtk_face_cell_points(const INT *face_index_tuple) {

  INT cell_tuple[3] = {0, 0, 0};
  this->get_mesh_tuple_owning_face_tuple(face_index_tuple, cell_tuple);
  auto cell_points = this->get_vtk_cell_points(cell_tuple);

  std::vector<double> points;
  points.reserve(3 * (1 << (this->ndim - 1)));
  const int face_index = static_cast<int>(face_index_tuple[0]);

  if (this->ndim == 2) {
    constexpr int indices[8] = {0, 1, 1, 2, 3, 2, 0, 3};

    for (int px = 0; px < 2; px++) {
      for (int cx = 0; cx < 3; cx++) {
        points.push_back(cell_points[3 * indices[face_index * 2 + px] + cx]);
      }
    }

  } else {

    /*
     *  Top
     *   7 - 6
     *   |   |
     *   4 - 5
     *  Bottom
     *
     *   3 - 2
     *   |   |
     *   0 - 1
     *
     *  y
     *  ^
     *  |
     *   -> x
     *
     */
    constexpr int indices[24] = {
        0, 1, 5, 4, // Face 0
        1, 2, 6, 5, // Face 1
        3, 2, 6, 7, // Face 2
        0, 3, 7, 4, // Face 3
        0, 1, 2, 3, // Face 4
        4, 5, 6, 7, // Face 5
    };

    for (int px = 0; px < 4; px++) {
      for (int cx = 0; cx < 3; cx++) {
        points.push_back(cell_points[3 * indices[face_index * 4 + px] + cx]);
      }
    }
  }

  return points;
}

std::vector<double>
CartesianHMesh::get_vtk_face_cell_points(const INT face_index) {
  NESOASSERT((0 <= face_index) &&
                 (face_index <
                  this->num_geoms_per_face_incscan[this->ndim == 2 ? 3 : 5]),
             "Bad face index passed.");

  INT face_index_tuple[3] = {0, 0, 0};
  this->get_face_id_as_tuple(face_index, face_index_tuple);
  return this->get_vtk_face_cell_points(face_index_tuple);
}

const std::vector<INT> &CartesianHMesh::get_owned_face_cells() {
  return this->owned_face_indices;
}

std::map<INT, VTK::UnstructuredCell> CartesianHMesh::get_vtk_face_cell_data() {
  std::map<INT, VTK::UnstructuredCell> output_data;

  for (auto ix : this->owned_face_indices) {

    VTK::UnstructuredCell tmp;

    tmp.num_points = this->ndim == 2 ? 2 : 4;
    tmp.points = this->get_vtk_face_cell_points(ix);
    tmp.cell_type =
        this->ndim == 2 ? VTK::CellType::line : VTK::CellType::quadrilateral;

    tmp.cell_data["face_cell_index"] = ix;
    output_data[ix] = tmp;
  }

  return output_data;
}

MPI_Comm CartesianHMesh::get_face_owning_ranks_comm() {
  return this->comm_faces;
}

} // namespace NESO::Particles
