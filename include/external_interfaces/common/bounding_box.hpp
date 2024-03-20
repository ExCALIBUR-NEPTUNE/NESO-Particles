#ifndef NESO_PARTICLES_BOUNDING_BOX_HPP_
#define NESO_PARTICLES_BOUNDING_BOX_HPP_

#include "../../typedefs.hpp"
#include <memory>
#include <vector>

namespace NESO::Particles::ExternalCommon {

/**
 * Container to store bounding boxes.
 */
class BoundingBox {
  std::vector<REAL> bb;

public:
  /**
   * Create a bounding box from a vector like:
   *  [min_x, min_y, min_z, max_x, max_y, max_z]
   * all entries must be present.
   *
   *  @param bb Bounding box.
   */
  BoundingBox(std::vector<REAL> &bb) : bb(bb) {
    NESOASSERT(bb.size() == 6,
               "Expected [min_x, min_y, min_z, max_x, max_y, max_z]");
  }

  /**
   * Return lower boundary in dimension.
   * @param dimx Dimension.
   * @returns Lower bound in dimension.
   */
  virtual inline REAL lower(const int dimx) const { return this->bb.at(dimx); }

  /**
   * Return upper boundary in dimension.
   * @param dimx Dimension.
   * @returns Upper bound in dimension.
   */
  virtual inline REAL upper(const int dimx) const {
    return this->bb.at(3 + dimx);
  }

  /**
   * Establish if a point is within the bounding box.
   *
   * @param ndim Number of dimensions to test.
   * @param point Point to test if in the bounding box.
   * @returns True if point in bounding box.
   */
  virtual inline bool contains_point(const int ndim,
                                     const std::vector<REAL> &point) const {
    bool is_contained = true;
    for (int dx = 0; dx < ndim; dx++) {
      is_contained = is_contained && ((this->lower(dx) <= point.at(dx)) &&
                                      (point.at(dx) <= this->upper(dx)));
    }
    return is_contained;
  }
};

typedef std::shared_ptr<BoundingBox> BoundingBoxSharedPtr;

} // namespace NESO::Particles::ExternalCommon

#endif
