#ifndef _NESO_PARTICLES_HIERARCHY
#define _NESO_PARTICLES_HIERARCHY
#include "compute_target.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <mpi.h>
#include <vector>

namespace NESO::Particles {

class MeshHierarchy {

public:
  MPI_Comm comm;
  int ndim;
  std::vector<int> dims;
  int subdivision_order;

  double cell_width_coarse;
  double cell_width_fine;
  double inverse_cell_width_coarse;
  double inverse_cell_width_fine;

  int ncells_coarse;
  int ncells_fine;

  MeshHierarchy(){};
  MeshHierarchy(MPI_Comm comm, const int ndim, std::vector<int> dims,
                const double extent = 1.0, const int subdivision_order = 1)
      : comm(comm), ndim(ndim), dims(dims),
        subdivision_order(subdivision_order), cell_width_coarse(extent),
        cell_width_fine(extent / ((double)std::pow(2, subdivision_order))),
        inverse_cell_width_coarse(1.0 / extent),
        inverse_cell_width_fine(((double)std::pow(2, subdivision_order)) /
                                extent),
        ncells_coarse(reduce_mul(ndim, dims)),
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
