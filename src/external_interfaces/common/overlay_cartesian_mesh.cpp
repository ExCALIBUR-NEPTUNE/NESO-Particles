#include <neso_particles/external_interfaces/common/overlay_cartesian_mesh.hpp>

namespace NESO::Particles::ExternalCommon {

int OverlayCartesianMesh::get_cell_in_dimension_lower(const int dim,
                                                      const REAL point) {
  NESOASSERT((dim > -1) && (dim < this->ndim), "Bad dimension passed.");

  const REAL shifted_point = point - this->origin.at(dim);
  REAL cell_float = shifted_point * this->inverse_cell_widths.at(dim);
  int cell = cell_float;

  const REAL cell_lower = static_cast<REAL>(cell) * this->cell_widths.at(dim);
  // move to the cell below if the point is exactly on the boundary.
  if (cell_lower >= (shifted_point - std::numeric_limits<REAL>::epsilon())) {
    cell--;
  }
  cell = (cell < 0) ? 0 : cell;
  cell = (cell >= this->cell_counts.at(dim)) ? this->cell_counts.at(dim) - 1
                                             : cell;
  return cell;
}

int OverlayCartesianMesh::get_cell_in_dimension_upper(const int dim,
                                                      const REAL point) {
  NESOASSERT((dim > -1) && (dim < this->ndim), "Bad dimension passed.");

  const REAL shifted_point = point - this->origin.at(dim);
  REAL cell_float = shifted_point * this->inverse_cell_widths.at(dim);
  int cell = cell_float;

  const REAL cell_upper =
      static_cast<REAL>(cell + 1) * this->cell_widths.at(dim);
  // move to the cell below if the point is exactly on the boundary.
  if (cell_upper <= (shifted_point + std::numeric_limits<REAL>::epsilon())) {
    cell++;
  }
  cell = (cell < 0) ? 0 : cell;
  cell = (cell >= this->cell_counts.at(dim)) ? this->cell_counts.at(dim) - 1
                                             : cell;
  return cell + 1;
}

void OverlayCartesianMesh::get_all_cells(int dim, std::vector<int> &cell_starts,
                                         std::vector<int> &cell_ends,
                                         std::vector<int> &index,
                                         std::vector<int> &cells) {

  for (int ix = cell_starts.at(dim); ix < cell_ends.at(dim); ix++) {
    index.at(dim) = ix;

    if (dim < (this->ndim - 1)) {
      get_all_cells(dim + 1, cell_starts, cell_ends, index, cells);
    } else {
      cells.push_back(this->get_linear_cell_index(index));
    }
  }
}

OverlayCartesianMesh::OverlayCartesianMesh(SYCLTargetSharedPtr sycl_target,
                                           const int ndim,
                                           const std::vector<REAL> &origin,
                                           const std::vector<REAL> &extents,
                                           const std::vector<int> &cell_counts)
    : sycl_target(sycl_target), ndim(ndim), origin(origin), extents(extents),
      cell_counts(cell_counts) {
  NESOASSERT(origin.size() == static_cast<std::size_t>(ndim),
             "Missmatch between origin.size and ndim.");
  NESOASSERT(extents.size() == static_cast<std::size_t>(ndim),
             "Missmatch between extents.size and ndim.");
  NESOASSERT(cell_counts.size() == static_cast<std::size_t>(ndim),
             "Missmatch between cell_counts.size and ndim.");

  this->cell_widths = std::vector<REAL>(ndim);
  this->inverse_cell_widths = std::vector<REAL>(ndim);

  for (int dimx = 0; dimx < this->ndim; dimx++) {
    NESOASSERT(this->extents.at(dimx) > 0.0,
               "Mesh has zero width in a dimension.");
    NESOASSERT(this->cell_counts.at(dimx) > 0,
               "Mesh has zero cell count in a dimension.");
    const REAL width =
        this->extents.at(dimx) / ((REAL)this->cell_counts.at(dimx));
    this->cell_widths.at(dimx) = width;
    this->inverse_cell_widths.at(dimx) = 1.0 / width;
  }

  this->d_origin =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, this->origin);
  this->d_extents =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, this->extents);
  this->d_cell_counts =
      std::make_shared<BufferDevice<int>>(this->sycl_target, this->cell_counts);
  this->d_cell_widths = std::make_shared<BufferDevice<REAL>>(this->sycl_target,
                                                             this->cell_widths);
  this->d_inverse_cell_widths = std::make_shared<BufferDevice<REAL>>(
      this->sycl_target, this->inverse_cell_widths);
}

OverlayCartesianMeshMapper OverlayCartesianMesh::get_device_mapper() {
  OverlayCartesianMeshMapper mapper;
  mapper.ndim = this->ndim;
  mapper.d_origin = this->d_origin->ptr;
  mapper.d_extents = this->d_extents->ptr;
  mapper.d_cell_counts = this->d_cell_counts->ptr;
  mapper.d_cell_widths = this->d_cell_widths->ptr;
  mapper.d_inverse_cell_widths = this->d_inverse_cell_widths->ptr;
  return mapper;
}

OverlayCartesianMeshMapper OverlayCartesianMesh::get_host_mapper() {
  OverlayCartesianMeshMapper mapper;
  mapper.ndim = this->ndim;
  mapper.d_origin = this->origin.data();
  mapper.d_extents = this->extents.data();
  mapper.d_cell_counts = this->cell_counts.data();
  mapper.d_cell_widths = this->cell_widths.data();
  mapper.d_inverse_cell_widths = this->inverse_cell_widths.data();
  return mapper;
}

int OverlayCartesianMesh::get_cell_in_dimension(const int dim,
                                                const REAL point) {
  NESOASSERT((dim > -1) && (dim < this->ndim), "Bad dimension passed.");
  NESOASSERT(point >= (this->origin.at(dim) - 1.0e-8),
             "Point is below lower bound.");
  NESOASSERT(point <= (this->origin.at(dim) + this->extents.at(dim) + 1.0e-8),
             "Point is above upper bound.");

  const REAL shifted_point = point - this->origin.at(dim);
  REAL cell_float = shifted_point * this->inverse_cell_widths.at(dim);
  int cell = cell_float;

  cell = (cell < 0) ? 0 : cell;
  cell = (cell >= this->cell_counts.at(dim)) ? this->cell_counts.at(dim) - 1
                                             : cell;
  return cell;
}

int OverlayCartesianMesh::get_cell_count() {
  int count = 1;
  for (int dimx = 0; dimx < this->ndim; dimx++) {
    count *= this->cell_counts.at(dimx);
  }
  return count;
}

void OverlayCartesianMesh::get_intersecting_cells(
    BoundingBoxSharedPtr bounding_box, std::vector<int> &cells) {
  cells.clear();
  std::vector<int> cell_starts;
  std::vector<int> cell_ends;

  int size = 1;
  for (int dimx = 0; dimx < this->ndim; dimx++) {
    const REAL bound_lower = bounding_box->lower(dimx);
    const int cell_lower = this->get_cell_in_dimension_lower(dimx, bound_lower);
    const REAL bound_upper = bounding_box->upper(dimx);
    const int cell_upper = this->get_cell_in_dimension_upper(dimx, bound_upper);
    cell_starts.push_back(cell_lower);
    cell_ends.push_back(cell_upper);
    size *= (cell_upper - cell_lower);
  }
  cells.reserve(size);
  std::vector<int> index(2);
  this->get_all_cells(0, cell_starts, cell_ends, index, cells);
}

std::shared_ptr<OverlayCartesianMesh>
create_overlay_mesh(SYCLTargetSharedPtr sycl_target, const int ndim,
                    BoundingBoxSharedPtr bounding_box, const int ncells) {
  std::vector<REAL> origin(ndim);
  std::vector<REAL> extents(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    origin.at(dimx) = bounding_box->lower(dimx);
    extents.at(dimx) = bounding_box->upper(dimx) - bounding_box->lower(dimx);
  }

  std::vector<int> cell_counts(ndim);
  std::fill(cell_counts.begin(), cell_counts.end(), 1);
  std::vector<REAL> widths(ndim);

  auto lambda_compute_widths = [&]() {
    for (int dimx = 0; dimx < ndim; dimx++) {
      widths.at(dimx) = extents.at(dimx) / cell_counts.at(dimx);
    }
  };
  lambda_compute_widths();

  auto lambda_compute_cell_count = [&]() -> int {
    int count = 1;
    for (int dimx = 0; dimx < ndim; dimx++) {
      count *= cell_counts.at(dimx);
    }
    return count;
  };

  while (lambda_compute_cell_count() < ncells) {
    // find dimension with widest cells
    auto max_iterator = std::max_element(widths.begin(), widths.end());
    auto max_location = std::distance(widths.begin(), max_iterator);
    // subdivide in dimension with widest cells.
    cell_counts.at(max_location) *= 2;
    lambda_compute_widths();
  }

  auto ocm = std::make_shared<OverlayCartesianMesh>(sycl_target, ndim, origin,
                                                    extents, cell_counts);
  return ocm;
}

} // namespace NESO::Particles::ExternalCommon
