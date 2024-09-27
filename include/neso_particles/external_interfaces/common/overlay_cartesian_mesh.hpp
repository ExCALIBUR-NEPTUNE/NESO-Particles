#ifndef _NESO_PARTICLES_OVERLAY_CARTESIAN_MESH_H_
#define _NESO_PARTICLES_OVERLAY_CARTESIAN_MESH_H_

#include "../../compute_target.hpp"
#include "bounding_box.hpp"

#include <memory>
#include <vector>

namespace NESO::Particles::ExternalCommon {

/**
 * Mapper for instances of @ref OverlayCartesianMesh. This type is designed to
 * be device copyable.
 */
struct OverlayCartesianMeshMapper {
  int ndim;
  REAL const *RESTRICT d_origin;
  REAL const *RESTRICT d_extents;
  int const *RESTRICT d_cell_counts;
  REAL const *RESTRICT d_cell_widths;
  REAL const *RESTRICT d_inverse_cell_widths;

  /**
   *  Return the cell which contains a point in the specified dimension.
   *
   *  @param[in] dim Dimension to find cell in.
   *  @param[in] point Coordinate in the requested dimension.
   *  @returns Containing cell in dimension.
   */
  inline int get_cell_in_dimension(const int dim, const REAL point) const {
    const REAL shifted_point = point - d_origin[dim];
    REAL cell_float = shifted_point * d_inverse_cell_widths[dim];
    int cell = cell_float;
    cell = (cell < 0) ? 0 : cell;
    cell = (cell >= d_cell_counts[dim]) ? d_cell_counts[dim] - 1 : cell;
    return cell;
  }

  /**
   *  Convert an index as a tuple to a linear index.
   *
   *  @param[in] cell_tuple Subscriptable host object with cell indices in each
   *  dimension.
   *  @returns Linear cell index.
   */
  template <typename T>
  inline int get_linear_cell_index(const T &cell_tuple) const {
    int idx = cell_tuple[this->ndim - 1];
    for (int dimx = (this->ndim - 2); dimx >= 0; dimx--) {
      idx *= d_cell_counts[dimx];
      idx += cell_tuple[dimx];
    }
    return idx;
  }

  /**
   * Validate a tuple cell index.
   *
   * @param index Cell tuple index to validate.
   * @returns True if cells is valid.
   */
  template <typename T>
  inline bool valid_linear_cell_index(const T &cell_tuple) const {
    bool valid = true;
    for (int dimx = 0; dimx < ndim; dimx++) {
      valid = valid && ((cell_tuple[dimx] > -1) &&
                        (cell_tuple[dimx] < d_cell_counts[dimx]));
    }
    return valid;
  }
};

/**
 * Generic n-D Cartesian mesh which can be placed over the locally owned
 * domain. For example if a coarse mesh is required to facilitate binning
 * particles into cells.
 */
class OverlayCartesianMesh {
protected:
  inline int get_cell_in_dimension_lower(const int dim, const REAL point) {
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

  inline int get_cell_in_dimension_upper(const int dim, const REAL point) {
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

  inline void get_all_cells(int dim, std::vector<int> &cell_starts,
                            std::vector<int> &cell_ends,
                            std::vector<int> &index, std::vector<int> &cells) {

    for (int ix = cell_starts.at(dim); ix < cell_ends.at(dim); ix++) {
      index.at(dim) = ix;

      if (dim < (this->ndim - 1)) {
        get_all_cells(dim + 1, cell_starts, cell_ends, index, cells);
      } else {
        cells.push_back(this->get_linear_cell_index(index));
      }
    }
  }

public:
  SYCLTargetSharedPtr sycl_target;
  int ndim;
  std::vector<REAL> origin;
  std::vector<REAL> extents;
  std::vector<int> cell_counts;
  std::vector<REAL> cell_widths;
  std::vector<REAL> inverse_cell_widths;

  std::shared_ptr<BufferDevice<REAL>> d_origin;
  std::shared_ptr<BufferDevice<REAL>> d_extents;
  std::shared_ptr<BufferDevice<int>> d_cell_counts;
  std::shared_ptr<BufferDevice<REAL>> d_cell_widths;
  std::shared_ptr<BufferDevice<REAL>> d_inverse_cell_widths;

  /**
   * Create an instance of a Cartesian mesh.
   *
   * @param sycl_target Compute device on which to place the mesh.
   * @param ndim Number of dimensions.
   * @param origin Origin for the mesh, must have size equal to the number of
   * dimensions.
   * @param extents Extents for the mesh, must have size equal to the number of
   * dimensions.
   * @param cell_counts Cell counts for the mesh, must have size equal to the
   * number of dimensions.
   */
  OverlayCartesianMesh(SYCLTargetSharedPtr sycl_target, const int ndim,
                       const std::vector<REAL> &origin,
                       const std::vector<REAL> &extents,
                       const std::vector<int> &cell_counts)
      : sycl_target(sycl_target), ndim(ndim), origin(origin), extents(extents),
        cell_counts(cell_counts) {
    NESOASSERT(origin.size() == ndim,
               "Missmatch between origin.size and ndim.");
    NESOASSERT(extents.size() == ndim,
               "Missmatch between extents.size and ndim.");
    NESOASSERT(cell_counts.size() == ndim,
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
    this->d_cell_counts = std::make_shared<BufferDevice<int>>(
        this->sycl_target, this->cell_counts);
    this->d_cell_widths = std::make_shared<BufferDevice<REAL>>(
        this->sycl_target, this->cell_widths);
    this->d_inverse_cell_widths = std::make_shared<BufferDevice<REAL>>(
        this->sycl_target, this->inverse_cell_widths);
  }

  /**
   * Get a mapper instance which is device copyable and which methods are only
   * callable on the device of the SYCL target.
   *
   * @returns Mapper instance only usable on the device.
   */
  inline OverlayCartesianMeshMapper get_device_mapper() {
    OverlayCartesianMeshMapper mapper;
    mapper.ndim = this->ndim;
    mapper.d_origin = this->d_origin->ptr;
    mapper.d_extents = this->d_extents->ptr;
    mapper.d_cell_counts = this->d_cell_counts->ptr;
    mapper.d_cell_widths = this->d_cell_widths->ptr;
    mapper.d_inverse_cell_widths = this->d_inverse_cell_widths->ptr;
    return mapper;
  }

  /**
   * Get a mapper instance which methods are only callable on the host.
   *
   * @returns Mapper instance only usable on the host.
   */
  inline OverlayCartesianMeshMapper get_host_mapper() {
    OverlayCartesianMeshMapper mapper;
    mapper.ndim = this->ndim;
    mapper.d_origin = this->origin.data();
    mapper.d_extents = this->extents.data();
    mapper.d_cell_counts = this->cell_counts.data();
    mapper.d_cell_widths = this->cell_widths.data();
    mapper.d_inverse_cell_widths = this->inverse_cell_widths.data();
    return mapper;
  }

  /**
   *  Return the cell which contains a point in the specified dimension.
   *
   *  @param[in] dim Dimension to find cell in.
   *  @param[in] point Coordinate in the requested dimension.
   *  @returns Containing cell in dimension.
   */
  inline int get_cell_in_dimension(const int dim, const REAL point) {
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

  /**
   *  Convert an index as a tuple to a linear index.
   *
   *  @param[in] cell_tuple Subscriptable host object with cell indices in each
   *  dimension.
   *  @returns Linear cell index.
   */
  template <typename T> inline int get_linear_cell_index(const T &cell_tuple) {
    int idx = cell_tuple[this->ndim - 1];
    for (int dimx = (this->ndim - 2); dimx >= 0; dimx--) {
      idx *= cell_counts.at(dimx);
      idx += cell_tuple.at(dimx);
    }
    return idx;
  }

  /**
   *  Get the number of cells in the mesh.
   */
  inline int get_cell_count() {
    int count = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      count *= this->cell_counts.at(dimx);
    }
    return count;
  }

  /**
   * Get a bounding box for a cell.
   *
   * @param[in] cell_tuple Subscriptable cell index tuple.
   * @param[out] bounding_box Bounding box.
   */
  template <typename T>
  inline BoundingBoxSharedPtr get_bounding_box(const T &cell_tuple) {
    std::vector<REAL> bounding_box(6);
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const int cell = cell_tuple[dimx];
      NESOASSERT(cell >= 0, "Bad cell index (below 0).");
      NESOASSERT(cell < cell_counts[dimx],
                 "Bad cell index (greater than cell count).");
      const REAL lb = origin.at(dimx) + cell * cell_widths.at(dimx);
      const REAL ub = origin.at(dimx) + (cell + 1) * cell_widths.at(dimx);
      bounding_box.at(dimx) = lb;
      bounding_box.at(dimx + 3) = ub;
    }
    return std::make_shared<BoundingBox>(bounding_box);
  }

  /**
   * Get the Cartesian cells which intersect a bounding box.
   *
   * @param[in] bounding_box Bounding box to find cells that intersect with.
   * @param[in, out] cells Output container for cells that insersect the box.
   */
  inline void get_intersecting_cells(BoundingBoxSharedPtr bounding_box,
                                     std::vector<int> &cells) {
    cells.clear();
    std::vector<int> cell_starts;
    std::vector<int> cell_ends;

    int size = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const REAL bound_lower = bounding_box->lower(dimx);
      const int cell_lower =
          this->get_cell_in_dimension_lower(dimx, bound_lower);
      const REAL bound_upper = bounding_box->upper(dimx);
      const int cell_upper =
          this->get_cell_in_dimension_upper(dimx, bound_upper);
      cell_starts.push_back(cell_lower);
      cell_ends.push_back(cell_upper);
      size *= (cell_upper - cell_lower);
    }
    cells.reserve(size);
    std::vector<int> index(2);
    this->get_all_cells(0, cell_starts, cell_ends, index, cells);
  }
};

/**
 * Create an overlay mesh from a bounding box and a minimum cell count.
 *
 * @param sycl_target Compute device for mesh.
 * @param ndim Number of dimensions.
 * @param bounding_box Bounding box to create mesh over.
 * @param ncells Minimum number of cells in mesh.
 * @returns Cartesian mesh overlaying bounding box with at least ncells cells.
 */
inline std::shared_ptr<OverlayCartesianMesh>
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

#endif
