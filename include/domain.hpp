#ifndef _NESO_PARTICLES_DOMAIN
#define _NESO_PARTICLES_DOMAIN
#include "mesh_hierarchy.hpp"
#include "typedefs.hpp"
#include <cstdint>
#include <cstdlib>
#include <mpi.h>

namespace NESO::Particles {

class HMesh {

public:
  virtual inline MPI_Comm get_comm() = 0;
  virtual inline int get_ndim() = 0;
  virtual inline std::vector<int> &get_dims() = 0;
  virtual inline int get_subdivision_order() = 0;
  virtual inline int get_cell_count() = 0;
  virtual inline double get_cell_width_coarse() = 0;
  virtual inline double get_cell_width_fine() = 0;
  virtual inline double get_inverse_cell_width_coarse() = 0;
  virtual inline double get_inverse_cell_width_fine() = 0;
  virtual inline int get_ncells_coarse() = 0;
  virtual inline int get_ncells_fine() = 0;
  virtual inline MeshHierarchy *get_mesh_hierarchy() = 0;
  virtual inline void free() = 0;
};

class CartesianHMesh : public HMesh {
private:
  int cell_count;
  MPI_Comm comm;
  MPI_Comm comm_cart;
  int periods[3] = {1, 1, 1};
  int coords[3] = {0, 0, 0};
  int mpi_dims[3] = {0, 0, 0};
  std::vector<int> cell_counts = {0, 0, 0};
  MeshHierarchy mesh_hierarchy;
  bool allocated = false;

public:
  int cell_starts[3] = {0, 0, 0};
  int cell_ends[3] = {1, 1, 1};
  int cell_counts_local[3] = {0, 0, 0};
  double global_extents[3] = {0.0, 0.0, 0.0};

  const int ndim;
  std::vector<int> &dims;
  const int subdivision_order;
  const double cell_width_coarse;
  const double cell_width_fine;
  const double inverse_cell_width_coarse;
  const double inverse_cell_width_fine;
  const int ncells_coarse;
  const int ncells_fine;

  CartesianHMesh(MPI_Comm comm, const int ndim, std::vector<int> &dims,
                 const double extent = 1.0, const int subdivision_order = 1)
      : comm(comm), ndim(ndim), dims(dims),
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
    // mpi_dims has monotonically decreasing order
    int rank, size;
    MPICHK(MPI_Comm_size(comm, &size));
    MPICHK(MPI_Comm_rank(comm, &rank));
    MPICHK(MPI_Dims_create(size, ndim, mpi_dims));

    for (int dimx = 0; dimx < ndim; dimx++) {
      cell_counts[dimx] = dims[dimx] * std::pow(2, subdivision_order);
      global_extents[dimx] = extent * dims[dimx];
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
    MPICHK(MPI_Cart_create(comm, ndim, mpi_dims_reordered.data(), periods, 1,
                           &comm_cart));
    this->allocated = true;

    // get the information about the cart comm that was actually created
    MPICHK(MPI_Cart_get(comm_cart, ndim, mpi_dims, periods, coords));

    // in each dimension compute the portion owned by this rank
    for (int dimx = 0; dimx < ndim; dimx++) {
      get_decomp_1d(mpi_dims[dimx], cell_counts[dimx], coords[dimx],
                    &cell_starts[dimx], &cell_ends[dimx]);
    }

    std::vector<double> origin(2);
    origin[0] = 0.0;
    origin[1] = 0.0;
    // for this mesh the hierarchy is simply a copy of the mesh
    this->mesh_hierarchy = MeshHierarchy(comm_cart, ndim, dims, origin,
                                         cell_width_coarse, subdivision_order);

    // setup the hierarchy

    // mesh tuple index
    INT index_mesh[3];
    // mesh_hierarchy tuple index
    INT index_mh[6];

    mesh_hierarchy.claim_initialise();

    // store the size of the local subdomain such that local linear indices can
    // be computed
    for (int dimx = 0; dimx < ndim; dimx++) {
      const int tmp_width = cell_ends[dimx] - cell_starts[dimx];
      NESOASSERT(tmp_width > 0,
                 "A domain of width <= 0 cells does not make sense");
      this->cell_counts_local[dimx] = tmp_width;
    }

    // loop over owned cells
    this->cell_count = 0;
    for (int cz = cell_starts[2]; cz < cell_ends[2]; cz++) {
      index_mesh[2] = cz;
      for (int cy = cell_starts[1]; cy < cell_ends[1]; cy++) {
        index_mesh[1] = cy;
        for (int cx = cell_starts[0]; cx < cell_ends[0]; cx++) {
          index_mesh[0] = cx;
          // convert mesh tuple index to mesh hierarchy tuple index
          mesh_tuple_to_mh_tuple(index_mesh, index_mh);
          // convert mesh hierarchy tuple index to global linear index in the
          // MeshHierarchy
          const INT index_global =
              mesh_hierarchy.tuple_to_linear_global(index_mh);
          // claim ownership of the current cell in the MeshHierarchy
          mesh_hierarchy.claim_cell(index_global, 1);
          this->cell_count++;
        }
      }
    }

    mesh_hierarchy.claim_finalise();
  };

  inline MPI_Comm get_comm() { return this->comm_cart; };
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
  inline int get_cell_count() { return this->cell_count; };
  inline MeshHierarchy *get_mesh_hierarchy() { return &this->mesh_hierarchy; };

  /*
   * Convert a mesh index (index_x, index_y, ...) for this cartesian mesh to
   * the format for a MeshHierarchy: (coarse_x, coarse_y,.., fine_x,
   * fine_y,...).
   */
  inline void mesh_tuple_to_mh_tuple(const INT *index_mesh, INT *index_mh) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div((long long)index_mesh[dimx],
                         (long long)mesh_hierarchy.ncells_dim_fine);
      index_mh[dimx] = pq.quot;
      index_mh[dimx + ndim] = pq.rem;
    }
  }

  inline void free() {
    int flag;
    MPICHK(MPI_Initialized(&flag))
    if (allocated && flag) {
      MPICHK(MPI_Comm_free(&this->comm_cart))
      this->comm_cart = MPI_COMM_NULL;
    }
    this->allocated = false;
    mesh_hierarchy.free();
  }
};

class Domain {
private:
public:
  HMesh &mesh;
  Domain(HMesh &mesh) : mesh(mesh) {}
  ~Domain() {}
};

} // namespace NESO::Particles

#endif
