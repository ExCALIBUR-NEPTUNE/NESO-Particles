#include <cstring>
#include <neso_particles/external_interfaces/common/bounding_box.hpp>

namespace NESO::Particles::ExternalCommon {

BoundingBox::BoundingBox() {
  std::fill(this->bb.begin(), this->bb.begin() + 3,
            std::numeric_limits<REAL>::max());
  std::fill(this->bb.begin() + 3, this->bb.begin() + 6,
            std::numeric_limits<REAL>::lowest());
}

BoundingBox::BoundingBox(std::vector<REAL> &bb) {
  NESOASSERT(bb.size() == 6,
             "Expected [min_x, min_y, min_z, max_x, max_y, max_z]");
  std::memcpy(this->bb.data(), bb.data(), 6 * sizeof(REAL));
}

void BoundingBox::BoundingBox::expand(
    std::shared_ptr<BoundingBox> bounding_box) {
  for (int dimx = 0; dimx < 3; dimx++) {
    this->bb.at(dimx) = std::min(this->bb.at(dimx), bounding_box->bb.at(dimx));
    this->bb.at(dimx + 3) =
        std::max(this->bb.at(dimx + 3), bounding_box->bb.at(dimx + 3));
  }
}

void BoundingBox::expand(const std::array<REAL, 3> &padding) {
  for (int dx = 0; dx < 3; dx++) {
    NESOASSERT(padding[dx] >= 0,
               "Negative padding passed in dimension: " + std::to_string(dx));
    this->bb[dx] -= padding[dx];
    this->bb[dx + 3] += padding[dx];
  }
}

REAL BoundingBox::lower(const int dimx) const { return this->bb.at(dimx); }

REAL BoundingBox::upper(const int dimx) const { return this->bb.at(3 + dimx); }

bool BoundingBox::contains_point(const int ndim,
                                 const std::vector<REAL> &point) const {
  bool is_contained = true;
  for (int dx = 0; dx < ndim; dx++) {
    is_contained = is_contained && ((this->lower(dx) <= point.at(dx)) &&
                                    (point.at(dx) <= this->upper(dx)));
  }
  return is_contained;
}

void BoundingBox::print() {
  nprint("x_direction, min: " + std::to_string(this->bb[0]) +
             ", max: " + std::to_string(this->bb[3]) + "\n",
         "y_direction, min: " + std::to_string(this->bb[1]) +
             ", max: " + std::to_string(this->bb[4]) + "\n",
         "z_direction, min: " + std::to_string(this->bb[2]) +
             ", max: " + std::to_string(this->bb[5]) + "\n");
}

} // namespace NESO::Particles::ExternalCommon
