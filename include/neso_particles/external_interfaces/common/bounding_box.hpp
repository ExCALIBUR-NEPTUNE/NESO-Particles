#ifndef _NESO_PARTICLES_BOUNDING_BOX_H_
#define _NESO_PARTICLES_BOUNDING_BOX_H_

#include "../../typedefs.hpp"
#include <array>
#include <memory>
#include <vector>

namespace NESO::Particles::ExternalCommon {

class BoundingBox;

/**
 * Container to store bounding boxes.
 */
class BoundingBox {
  std::array<REAL, 6> bb;

public:
  BoundingBox();
  virtual ~BoundingBox() = default;

  /**
   * Create a bounding box from a vector like:
   *  [min_x, min_y, min_z, max_x, max_y, max_z]
   * all entries must be present.
   *
   *  @param bb Bounding box.
   */
  BoundingBox(std::vector<REAL> &bb);

  /**
   * Expand a bounding box to encompass another bounding box.
   *
   * @param bounding_box Bounding box to include in this bounding box.
   */
  void expand(std::shared_ptr<BoundingBox> bounding_box);

  /**
   * Expand a bounding box by the specified padding in each dimension.
   *
   * @param padding Padding to apply in each direction.
   */
  void expand(const std::array<REAL, 3> &padding);

  /**
   * Return lower boundary in dimension.
   * @param dimx Dimension.
   * @returns Lower bound in dimension.
   */
  virtual REAL lower(const int dimx) const;

  /**
   * Return upper boundary in dimension.
   * @param dimx Dimension.
   * @returns Upper bound in dimension.
   */
  virtual REAL upper(const int dimx) const;

  /**
   * Establish if a point is within the bounding box.
   *
   * @param ndim Number of dimensions to test.
   * @param point Point to test if in the bounding box.
   * @returns True if point in bounding box.
   */
  virtual bool contains_point(const int ndim,
                              const std::vector<REAL> &point) const;

  /**
   * Print the bounding box information on stdout.
   */
  void print();
};

typedef std::shared_ptr<BoundingBox> BoundingBoxSharedPtr;

} // namespace NESO::Particles::ExternalCommon

#endif
