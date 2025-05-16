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

} // namespace NESO::Particles
