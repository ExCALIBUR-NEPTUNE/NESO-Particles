#ifndef _NESO_PARTICLES_HIERARCHY
#define _NESO_PARTICLES_HIERARCHY
#include <cmath>
#include <vector>

#include "compute_target.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class MeshHierarchy {

public:
  SYCLTarget &sycl_target;
  const int ndim;
  std::vector<int> &dims;
  const int subdivision_order;

  const double cell_width_coarse;
  const double cell_width_fine;
  const double inverse_cell_width_coarse;
  const double inverse_cell_width_fine;

  const int ncells_coarse;
  const int ncells_fine;

  MeshHierarchy(SYCLTarget &sycl_target, const int ndim, std::vector<int> &dims,
                const double extent = 1.0, const int subdivision_order = 1)
      : sycl_target(sycl_target), ndim(ndim), dims(dims),
        subdivision_order(subdivision_order), cell_width_coarse(extent),
        cell_width_fine(extent / ((double)std::pow(2, subdivision_order))),
        inverse_cell_width_coarse(1.0 / extent),
        inverse_cell_width_fine(((double)std::pow(2, subdivision_order)) /
                                extent),
        ncells_coarse(std::accumulate(dims.begin(), dims.begin() + ndim, 1,
                                      std::multiplies<>{})),
        ncells_fine(std::pow(std::pow(2, subdivision_order), ndim)) {
    NESOASSERT(dims.size() >= ndim, "vector of dims too small");
    for (int dimx = 0; dimx < ndim; dimx++) {
      NESOASSERT(dims[dimx] > 0, "Dim size is <= 0 in a direction.");
    }
    NESOASSERT(cell_width_coarse > 0.0, "Extent <= 0.0 passed");
    NESOASSERT(subdivision_order >= 0, "Negative subdivision order passed.");
  };
};

} // namespace NESO::Particles

#endif
