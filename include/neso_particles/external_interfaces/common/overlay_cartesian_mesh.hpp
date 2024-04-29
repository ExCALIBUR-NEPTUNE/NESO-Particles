#ifndef _NESO_PARTICLES_OVERLAY_CARTESIAN_MESH_H_
#define _NESO_PARTICLES_OVERLAY_CARTESIAN_MESH_H_

#include "../../compute_target.hpp"
#include "bounding_box.hpp"

#include <memory>
#include <vector>

namespace NESO::Particles::ExternalCommon {

/**
 * TODO
 */
class OverlayCartesianMesh {
protected:
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
   * TODO
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
   *  Return the cell which contains a point in the specified dimension.
   *
   *  @param[in] dim Dimension to find cell in.
   *  @param[in] point Coordinate in the requested dimension.
   *  @returns Containing cell in dimension.
   */
  inline int get_cell_in_dimension(const int dim, const double point) {
    NESOASSERT((dim > -1) && (dim < this->ndim), "Bad dimension passed.");
    NESOASSERT(point >= (this->origin.at(dim) - 1.0e-8),
               "Point is below lower bound.");
    NESOASSERT(point <= (this->origin.at(dim) + this->extents.at(dim) + 1.0e-8),
               "Point is above upper bound.");

    const double shifted_point = point - this->origin.at(dim);
    double cell_float = shifted_point * this->inverse_cell_widths.at(dim);
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
      const double lb = origin.at(dimx) + cell * cell_widths.at(dimx);
      const double ub = origin.at(dimx) + (cell + 1) * cell_widths.at(dimx);
      bounding_box.at(dimx) = lb;
      bounding_box.at(dimx + 3) = ub;
    }
    return std::make_shared<BoundingBox>(bounding_box);
  }
};

} // namespace NESO::Particles::ExternalCommon

#endif
