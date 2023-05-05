#ifndef _NESO_PARTICLES_MESH_INTERFACE
#define _NESO_PARTICLES_MESH_INTERFACE
#include "mesh_hierarchy.hpp"
#include "typedefs.hpp"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <set>
#include <vector>

namespace NESO::Particles {

/**
 *  Abstract base class for mesh types over which a MeshHierarchy is placed.
 */
class HMesh {

public:
  /**
   * Get the MPI communicator of the mesh.
   *
   * @returns MPI communicator.
   */
  virtual inline MPI_Comm get_comm() = 0;
  /**
   *  Get the number of dimensions of the mesh.
   *
   *  @returns Number of mesh dimensions.
   */
  virtual inline int get_ndim() = 0;
  /**
   *  Get the Mesh dimensions.
   *
   *  @returns Mesh dimensions.
   */
  virtual inline std::vector<int> &get_dims() = 0;
  /**
   * Get the subdivision order of the mesh.
   *
   * @returns Subdivision order.
   */
  virtual inline int get_subdivision_order() = 0;
  /**
   * Get the total number of cells in the mesh.
   *
   * @returns Total number of mesh cells.
   */
  virtual inline int get_cell_count() = 0;
  /**
   * Get the mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy coarse cell width.
   */
  virtual inline double get_cell_width_coarse() = 0;
  /**
   * Get the mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy fine cell width.
   */
  virtual inline double get_cell_width_fine() = 0;
  /**
   * Get the inverse mesh width of the coarse cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse coarse cell width.
   */
  virtual inline double get_inverse_cell_width_coarse() = 0;
  /**
   * Get the inverse mesh width of the fine cells in the MeshHierarchy.
   *
   * @returns MeshHierarchy inverse fine cell width.
   */
  virtual inline double get_inverse_cell_width_fine() = 0;
  /**
   *  Get the global number of coarse cells.
   *
   *  @returns Global number of coarse cells.
   */
  virtual inline int get_ncells_coarse() = 0;
  /**
   *  Get the number of fine cells per coarse cell.
   *
   *  @returns Number of fine cells per coarse cell.
   */
  virtual inline int get_ncells_fine() = 0;
  /**
   * Get the MeshHierarchy instance placed over the mesh.
   *
   * @returns MeshHierarchy placed over the mesh.
   */
  virtual inline std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() = 0;
  /**
   *  Free the mesh and associated communicators.
   */
  virtual inline void free() = 0;
  /**
   *  Get a std::vector of MPI ranks which should be used to setup local
   *  communication patterns.
   *
   *  @returns std::vector of MPI ranks.
   */
  virtual inline std::vector<int> &get_local_communication_neighbours() = 0;
  /**
   *  Get a point in the domain that should be in, or at least close to, the
   *  sub-domain on this MPI process. Useful for parallel initialisation.
   *
   *  @param point Pointer to array of size equal to at least the number of mesh
   * dimensions.
   */
  virtual inline void get_point_in_subdomain(double *point) = 0;
};

typedef std::shared_ptr<HMesh> HMeshSharedPtr;

/**
 * Example mesh that duplicates a MeshHierarchy as a HMesh for examples and
 * testing.
 */
class CartesianHMesh : public HMesh {
private:
  int cell_count;
  MPI_Comm comm;
  MPI_Comm comm_cart;
  int periods[3] = {1, 1, 1};
  int coords[3] = {0, 0, 0};
  int mpi_dims[3] = {0, 0, 0};
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;
  bool allocated = false;

  std::vector<int> neighbour_ranks;

public:
  /// Disable (implicit) copies.
  CartesianHMesh(const CartesianHMesh &st) = delete;
  /// Disable (implicit) copies.
  CartesianHMesh &operator=(CartesianHMesh const &a) = delete;

  /// Holds the first cell this rank owns in each dimension.
  int cell_starts[3] = {0, 0, 0};
  /// Holds the last cell+1 this ranks owns in each dimension.
  int cell_ends[3] = {1, 1, 1};
  /// Global number of cells in each dimension.
  std::vector<int> cell_counts = {0, 0, 0};
  /// Local number of cells in each dimension.
  int cell_counts_local[3] = {0, 0, 0};
  /// Global extents of the mesh.
  double global_extents[3] = {0.0, 0.0, 0.0};
  /// Width of the stencil used to determine which MPI ranks are neighbours.
  int stencil_width;
  /// Number of dimensions of the mesh.
  const int ndim;
  /// Vector holding the number of coarse cells in each dimension.
  std::vector<int> &dims;
  /// Subdivision order to determine number of fine cells per coarse cell.
  const int subdivision_order;
  /// Width of coarse cells, uniform in all dimensions.
  const double cell_width_coarse;
  /// Width of fine cells, uniform in all dimensions.
  const double cell_width_fine;
  /// Inverse of the coarse cell width.
  const double inverse_cell_width_coarse;
  /// Inverse of the fine cell width.
  const double inverse_cell_width_fine;
  /// Global number of coarse cells.
  const int ncells_coarse;
  /// Number of coarse cells per fine cell.
  const int ncells_fine;
  /**
   * Construct a mesh over a given MPI communicator with a specified shape.
   *
   * @param comm MPI Communicator to use for decomposition.
   * @param ndim Number of dimensions.
   * @param dims Number of coarse cells in each dimension.
   * @param extent Width of each coarse cell in each dimension.
   * @param subdivision_order Number of times to subdivide each coarse cell to
   * produce the fine cells.
   * @param stencil_width Width of the stencil, in number of cells, used to
   * determine MPI neighbours.
   */
  CartesianHMesh(MPI_Comm comm, const int ndim, std::vector<int> &dims,
                 const double extent = 1.0, const int subdivision_order = 1,
                 const int stencil_width = 0)
      : comm(comm), ndim(ndim), dims(dims),
        subdivision_order(subdivision_order), stencil_width(stencil_width),
        cell_width_coarse(extent),
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

    std::vector<double> origin(ndim);
    for (int dimx = 0; dimx < ndim; dimx++) {
      origin[dimx] = 0.0;
    }
    // for this mesh the hierarchy is simply a copy of the mesh
    this->mesh_hierarchy = std::make_shared<MeshHierarchy>(
        comm_cart, ndim, dims, origin, cell_width_coarse, subdivision_order);

    // setup the hierarchy

    // mesh tuple index
    INT index_mesh[3];
    // mesh_hierarchy tuple index
    INT index_mh[6];

    mesh_hierarchy->claim_initialise();

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
              mesh_hierarchy->tuple_to_linear_global(index_mh);
          // claim ownership of the current cell in the MeshHierarchy
          mesh_hierarchy->claim_cell(index_global, 1);
          this->cell_count++;
        }
      }
    }

    mesh_hierarchy->claim_finalise();

    // get the MPI ranks owning cells within the stencil region

    int stencil_cell_starts[3] = {0, 0, 0};
    int stencil_cell_ends[3] = {1, 1, 1};

    for (int dimx = 0; dimx < this->ndim; dimx++) {
      stencil_cell_starts[dimx] = this->cell_starts[dimx] - this->stencil_width;
      stencil_cell_ends[dimx] = this->cell_ends[dimx] + this->stencil_width;
    }

    std::set<int> neighbour_ranks_set;
    for (int scz = stencil_cell_starts[2]; scz < stencil_cell_ends[2]; scz++) {
      const int cz = (scz + this->stencil_width * this->cell_counts[2]) %
                     MAX(this->cell_counts[2], 1);
      index_mesh[2] = cz;
      for (int scy = stencil_cell_starts[1]; scy < stencil_cell_ends[1];
           scy++) {
        const int cy = (scy + this->stencil_width * this->cell_counts[1]) %
                       MAX(this->cell_counts[1], 1);
        index_mesh[1] = cy;
        for (int scx = stencil_cell_starts[0]; scx < stencil_cell_ends[0];
             scx++) {
          const int cx = (scx + this->stencil_width * this->cell_counts[0]) %
                         this->cell_counts[0];
          index_mesh[0] = cx;
          // convert mesh tuple index to mesh hierarchy tuple index
          mesh_tuple_to_mh_tuple(index_mesh, index_mh);
          // convert mesh hierarchy tuple index to global linear index in the
          // MeshHierarchy
          const INT index_global =
              mesh_hierarchy->tuple_to_linear_global(index_mh);
          // claim ownership of the current cell in the MeshHierarchy
          const int owning_rank = this->mesh_hierarchy->get_owner(index_global);
          neighbour_ranks_set.insert(owning_rank);
        }
      }
    }

    this->neighbour_ranks.reserve(neighbour_ranks_set.size());
    for (auto rankx : neighbour_ranks_set) {
      this->neighbour_ranks.push_back(rankx);
    }
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
  inline std::shared_ptr<MeshHierarchy> get_mesh_hierarchy() {
    return this->mesh_hierarchy;
  };

  /**
   * Convert a mesh index (index_x, index_y, ...) for this cartesian mesh to
   * the format for a MeshHierarchy: (coarse_x, coarse_y,.., fine_x,
   * fine_y,...).
   *
   * @param index_mesh Input tuple index on mesh.
   * @param index_mh Output tuple index on MeshHierarchy.
   */
  inline void mesh_tuple_to_mh_tuple(const INT *index_mesh, INT *index_mh) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div((long long)index_mesh[dimx],
                         (long long)mesh_hierarchy->ncells_dim_fine);
      index_mh[dimx] = pq.quot;
      index_mh[dimx + ndim] = pq.rem;
    }
  }

  /**
   *  Free the mesh and any associated communicators.
   */
  inline void free() {
    int flag;
    MPICHK(MPI_Initialized(&flag))
    if (allocated && flag) {
      MPICHK(MPI_Comm_free(&this->comm_cart))
      this->comm_cart = MPI_COMM_NULL;
    }
    this->allocated = false;
    mesh_hierarchy->free();
  }

  inline std::vector<int> &get_local_communication_neighbours() {
    return this->neighbour_ranks;
  }

  inline void get_point_in_subdomain(double *point) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const double start = this->cell_starts[dimx] * this->cell_width_fine;
      const double end = this->cell_ends[dimx] * this->cell_width_fine;
      point[dimx] = 0.5 * (start + end);
    }
  };
};

typedef std::shared_ptr<CartesianHMesh> CartesianHMeshSharedPtr;

} // namespace NESO::Particles

#endif
