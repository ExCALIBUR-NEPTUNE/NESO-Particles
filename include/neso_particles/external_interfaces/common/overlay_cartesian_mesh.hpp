#ifndef _NESO_PARTICLES_OVERLAY_CARTESIAN_MESH_H_
#define _NESO_PARTICLES_OVERLAY_CARTESIAN_MESH_H_

#include "../../compute_target.hpp"
#include "../../device_buffers.hpp"
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
  int get_cell_in_dimension_lower(const int dim, const REAL point);

  int get_cell_in_dimension_upper(const int dim, const REAL point);

  void get_all_cells(int dim, std::vector<int> &cell_starts,
                     std::vector<int> &cell_ends, std::vector<int> &index,
                     std::vector<int> &cells);

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
                       const std::vector<int> &cell_counts);

  /**
   * Get a mapper instance which is device copyable and which methods are only
   * callable on the device of the SYCL target.
   *
   * @returns Mapper instance only usable on the device.
   */
  OverlayCartesianMeshMapper get_device_mapper();

  /**
   * Get a mapper instance which methods are only callable on the host.
   *
   * @returns Mapper instance only usable on the host.
   */
  OverlayCartesianMeshMapper get_host_mapper();

  /**
   *  Return the cell which contains a point in the specified dimension.
   *
   *  @param[in] dim Dimension to find cell in.
   *  @param[in] point Coordinate in the requested dimension.
   *  @returns Containing cell in dimension.
   */
  int get_cell_in_dimension(const int dim, const REAL point);

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
  int get_cell_count();

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
  void get_intersecting_cells(BoundingBoxSharedPtr bounding_box,
                              std::vector<int> &cells);
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
std::shared_ptr<OverlayCartesianMesh>
create_overlay_mesh(SYCLTargetSharedPtr sycl_target, const int ndim,
                    BoundingBoxSharedPtr bounding_box, const int ncells);

} // namespace NESO::Particles::ExternalCommon

#endif
