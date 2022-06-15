#ifndef _NESO_PARTICLES_HIERARCHY
#define _NESO_PARTICLES_HIERARCHY
#include "compute_target.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <mpi.h>
#include <vector>

namespace NESO::Particles {

class MeshHierarchy {

private:
  int ncell_dim_fine;

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
    ncell_dim_fine = std::pow(2, subdivision_order);
  };

  /*
   * tuple should be:
   * 1D: (coarse_x, fine_x)
   * 2D: (coarse_x, coarse_y, fine_x, fine_y)
   * 3D: (coarse_x, coarse_y, coarse_z, fine_x, fine_y, fine_z)
   */
  inline int tuple_to_linear_global(int *index_tuple) {
    int index_coarse = tuple_to_linear_coarse(index_tuple);
    int index_fine = tuple_to_linear_fine(&index_tuple[ndim]);
    int index = index_coarse * ncells_fine + index_fine;
    return index;
  };
  inline int tuple_to_linear_coarse(int *index_tuple) {
    int index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= dims[dimx];
      index += index_tuple[dimx];
    }
    return index;
  };
  inline int tuple_to_linear_fine(int *index_tuple) {
    int index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= ncell_dim_fine;
      index += index_tuple[dimx];
    }
    return index;
  };
  inline void linear_to_tuple_global(int linear, int *index) {
    auto pq = std::div(linear, ncells_fine);
    linear_to_tuple_coarse(pq.quot, index);
    linear_to_tuple_fine(pq.rem, index + ndim);
  };
  inline void linear_to_tuple_coarse(int linear, int *index) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div(linear, dims[dimx]);
      index[dimx] = pq.rem;
      linear = pq.quot;
    }
  };
  inline void linear_to_tuple_fine(int linear, int *index) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div(linear, ncell_dim_fine);
      index[dimx] = pq.rem;
      linear = pq.quot;
    }
  };
};

} // namespace NESO::Particles

#endif
