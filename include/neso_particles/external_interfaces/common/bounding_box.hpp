#ifndef _NESO_PARTICLES_BOUNDING_BOX_H_
#define _NESO_PARTICLES_BOUNDING_BOX_H_

#include "../../typedefs.hpp"
#include <memory>
#include <vector>

namespace NESO::Particles::ExternalCommon {

class BoundingBox;

/**
 * Container to store bounding boxes.
 */
class BoundingBox {
  std::vector<REAL> bb;

public:
  BoundingBox() {
    this->bb.resize(6);
    std::fill(this->bb.begin(), this->bb.begin() + 3,
              std::numeric_limits<REAL>::max());
    std::fill(this->bb.begin() + 3, this->bb.begin() + 6,
              std::numeric_limits<REAL>::lowest());
  }
  virtual ~BoundingBox() = default;

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
   * Expand a bounding box to encompass another bounding box.
   *
   * @param bounding_box Bounding box to include in this bounding box.
   */
  inline void expand(std::shared_ptr<BoundingBox> bounding_box);

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

  /**
   * Print the bounding box information on stdout.
   */
  inline void print() {
    nprint("x_direction, min: " + std::to_string(this->bb[0]) +
               ", max: " + std::to_string(this->bb[3]) + "\n",
           "y_direction, min: " + std::to_string(this->bb[1]) +
               ", max: " + std::to_string(this->bb[4]) + "\n",
           "z_direction, min: " + std::to_string(this->bb[2]) +
               ", max: " + std::to_string(this->bb[5]) + "\n");
  }
};

/**
 * Expand a bounding box to encompass another bounding box.
 *
 * @param bounding_box Bounding box to include in this bounding box.
 */
inline void BoundingBox::expand(std::shared_ptr<BoundingBox> bounding_box) {
  for (int dimx = 0; dimx < 3; dimx++) {
    this->bb.at(dimx) = std::min(this->bb.at(dimx), bounding_box->bb.at(dimx));
    this->bb.at(dimx + 3) =
        std::max(this->bb.at(dimx + 3), bounding_box->bb.at(dimx + 3));
  }
}

typedef std::shared_ptr<BoundingBox> BoundingBoxSharedPtr;

} // namespace NESO::Particles::ExternalCommon

#endif
