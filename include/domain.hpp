#ifndef _NESO_PARTICLES_DOMAIN
#define _NESO_PARTICLES_DOMAIN
#include "typedefs.hpp"
#include <mpi.h>

namespace NESO::Particles {

class Mesh {
private:
  int cell_count;

public:
  Mesh(){};
  Mesh(int cell_count) : cell_count(cell_count){};
  inline int get_cell_count() { return this->cell_count; };
};

class HMesh : public Mesh {

private:
public:
  virtual inline SYCLTarget &get_sycl_target() = 0;
  virtual inline int get_ndim() = 0;
  virtual inline std::vector<int> &get_dims() = 0;
  virtual inline int get_subdivision_order() = 0;
  virtual inline double get_cell_width_coarse() = 0;
  virtual inline double get_cell_width_fine() = 0;
  virtual inline double get_inverse_cell_width_coarse() = 0;
  virtual inline double get_inverse_cell_width_fine() = 0;
  virtual inline int get_ncells_coarse() = 0;
  virtual inline int get_ncells_fine() = 0;
};

class CartesianHMesh : public HMesh {
private:
  int cell_count;
  MPI_Comm comm_cart;

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

  CartesianHMesh(SYCLTarget &sycl_target, const int ndim,
                 std::vector<int> &dims, const double extent = 1.0,
                 const int subdivision_order = 1)
      : sycl_target(sycl_target), ndim(ndim), dims(dims),
        subdivision_order(subdivision_order), cell_width_coarse(extent),
        cell_width_fine(extent / ((double)std::pow(2, subdivision_order))),
        inverse_cell_width_coarse(1.0 / extent),
        inverse_cell_width_fine(((double)std::pow(2, subdivision_order)) /
                                extent),
        ncells_coarse(reduce_mul(ndim, dims)),
        ncells_fine(std::pow(std::pow(2, subdivision_order), ndim)) {

    // basic error checking of inputs
    NESOASSERT(dims.size() >= ndim, "vector of dims too small");
    for (int dimx = 0; dimx < ndim; dimx++) {
      NESOASSERT(dims[dimx] > 0, "Dim size is <= 0 in a direction.");
    }
    NESOASSERT(cell_width_coarse > 0.0, "Extent <= 0.0 passed");
    NESOASSERT(subdivision_order >= 0, "Negative subdivision order passed.");

    // mpi decompose
    std::vector<int> mpi_dims = {0, 0, 0};
    // mpi_dims has monotonically decreasing order
    MPICHK(MPI_Dims_create(sycl_target.comm_pair.size_parent, ndim,
                           mpi_dims.data()));

    std::vector<int> cell_counts(ndim);
    for (int dimx = 0; dimx < ndim; dimx++) {
      cell_counts[dimx] = dims[dimx] * std::pow(2, subdivision_order);
    }
    // direction with most cells first to match mpi_dims order
    auto cell_count_ordering = reverse_argsort(cell_counts);

    // reorder the mpi_dims to match the actual domain
    std::vector<int> mpi_dims_reordered(ndim);
    for (int dimx = 0; dimx < ndim; dimx++) {
      mpi_dims_reordered[cell_count_ordering[dimx]] = mpi_dims[dimx];
    }

    // create MPI cart comm with a decomposition that roughly makes sense for
    // the shape of the domain
    int periods[3] = {1, 1, 1};
    MPICHK(MPI_Cart_create(sycl_target.comm, ndim, mpi_dims_reordered.data(),
                           periods, 1, &comm_cart));
  };

  virtual inline SYCLTarget &get_sycl_target() { return this->sycl_target; };
  inline int get_ndim() { return this->ndim; };
  inline std::vector<int> &get_dims() { return this->dims; };
  inline int get_subdivision_order() { return this->subdivision_order; };
  inline double get_cell_width_coarse() { return this->cell_width_coarse; };
  inline double get_cell_width_fine() { return this->cell_width_fine; };
  inline double get_inverse_cell_width_coarse() {
    return this->inverse_cell_width_coarse;
  };
  inline double get_inverse_cell_width_fine() {
    return this->inverse_cell_width_fine;
  };
  inline int get_ncells_coarse() { return this->ncells_coarse; };
  inline int get_ncells_fine() { return this->ncells_fine; };
};

class Domain {
private:
public:
  Mesh &mesh;
  Domain(Mesh &mesh) : mesh(mesh) {}
  ~Domain() {}
};

} // namespace NESO::Particles

#endif
