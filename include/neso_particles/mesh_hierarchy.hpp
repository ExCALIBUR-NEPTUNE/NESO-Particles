#ifndef _NESO_PARTICLES_HIERARCHY
#define _NESO_PARTICLES_HIERARCHY
#include "communication.hpp"
#include "communication/global_move_communication.hpp"
#include "compute_target.hpp"
#include "device_buffers.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <stack>
#include <utility>
#include <vector>

namespace NESO::Particles {

const int mask = std::numeric_limits<int>::lowest();

/**
 * Structure to place a coarse mesh over an arbitrary shaped domain. Each cell
 * in the coarse mesh is a cubes/square that contains a fine mesh also of
 * cubes/squares. Each cell is indexed with a tuple for the coarse mesh and a
 * tuple for the fine mesh in the coarse cell. Each cell in the fine mesh is
 * owned by a unique MPI rank.
 *
 * This class contains the description of the coarse and fine meshes and the
 * map from fine cells to owning MPI rank. The map is accessible on all MPI
 * ranks and is stored once per MPI shared memory region.
 */
class MeshHierarchy {

private:
  std::stack<std::pair<INT, int>> claim_stack;
  std::stack<std::pair<INT, int>> claim_stack_binned;

  MPI_Win map_win;
  bool map_allocated = false;
  int *map = NULL;
  int *map_base = NULL;
  bool map_created = false;

  inline void all_reduce_max_map() {
    MPICHK(MPI_Win_fence(0, this->map_win))
    if (comm_pair.rank_intra == 0) {
      int *tmp_map = (int *)std::malloc(this->ncells_global * sizeof(int));
      NESOASSERT(tmp_map != NULL, "malloc failed");

      for (int cellx = 0; cellx < this->ncells_global; cellx++) {
        tmp_map[cellx] = this->map[cellx];
      }

      MPICHK(MPI_Allreduce(tmp_map, this->map, this->ncells_global, MPI_INT,
                           MPI_MAX, comm_pair.comm_inter))

      std::free(tmp_map);
    }
    MPICHK(MPI_Win_fence(0, this->map_win))
  };

public:
  /// Disable (implicit) copies.
  MeshHierarchy(const MeshHierarchy &st) = delete;
  /// Disable (implicit) copies.
  MeshHierarchy &operator=(MeshHierarchy const &a) = delete;

  /// MPI communicator on which this mesh is decomposed.
  MPI_Comm comm;
  /// CommPair instance that contains the inter and intra communicators.
  CommPair comm_pair;
  /// The container for MPI objects required for global particle movement.
  GlobalMoveCommunicationSharedPtr global_move_communication;

  /// Number of mesh dimensions (1,2 or 3).
  int ndim;
  /// Number of coarse cells in each dimension.
  std::vector<int> dims;
  /// Origin of the coarse mesh in each dimension.
  std::vector<double> origin;
  /// The fine mesh cells have width `cell_width_coarse / 2^p` for subdivision
  /// order p.
  int subdivision_order;
  /// Cell width of the coarse cells.
  double cell_width_coarse;
  /// Cell width of the fine cells.
  double cell_width_fine;
  /// `1/cell_width_coarse`.
  double inverse_cell_width_coarse;
  /// `1/cell_width_fine`.
  double inverse_cell_width_fine;

  /// Global number of coarse cells.
  INT ncells_coarse;
  /// Number of fine cells in each coarse cell.
  INT ncells_fine;
  /// Number of fine cells per dimension in each coarse cell.
  INT ncells_dim_fine;
  /// Global number of fine cells.
  INT ncells_global;

  /**
   * Prints information about the MeshHierarchy to stdout.
   */
  inline void print() {
    nprint("-------------------------------------------------------------------"
           "-------------");
    nprint("                        MeshHierarchy Infomation");
    nprint("ndim:", ndim);
    std::cout << "dims: ";
    for (int dx = 0; dx < ndim; dx++) {
      std::cout << dims.at(dx) << " ";
    }
    std::cout << std::endl;
    std::cout << "origin: ";
    for (int dx = 0; dx < ndim; dx++) {
      std::cout << origin.at(dx) << " ";
    }
    std::cout << std::endl;
    nprint("subdivision_order:", subdivision_order);
    nprint("cell_width_coarse:", cell_width_coarse);
    nprint("cell_width_fine:", cell_width_fine);
    nprint("-------------------------------------------------------------------"
           "-------------");
  }

  MeshHierarchy(){};

  /**
   *  Create a mesh hierarchy decomposed over the given MPI communicator with a
   *  specified number and shape of coarse cells and subdivision order. Must be
   *  called collectively on the communicator.
   *
   *  @param comm MPI communicator to use for decomposition.
   *  @param ndim Number of dimensions (1,2 or 3).
   *  @param dims Number of coarse cells in each dimension.
   *  @param origin Origin to use for mesh.
   *  @param extent Cell width of a coarse cell in each dimension.
   *  @param subdivision_order Number of times to subdivide each coarse cell in
   *  each dimension to produce the fine mesh.
   *
   */
  MeshHierarchy(MPI_Comm comm, const int ndim, std::vector<int> dims,
                std::vector<double> origin, const double extent = 1.0,
                const int subdivision_order = 1)
      : comm(comm), comm_pair(comm),
        global_move_communication(
            std::make_shared<GlobalMoveCommunication>(comm)),
        ndim(ndim), dims(dims), origin(origin),
        subdivision_order(subdivision_order), cell_width_coarse(extent),
        cell_width_fine(extent / ((double)std::pow(2, subdivision_order))),
        inverse_cell_width_coarse(1.0 / extent),
        inverse_cell_width_fine(((double)std::pow(2, subdivision_order)) /
                                extent),
        ncells_coarse(reduce_mul(ndim, dims)),
        ncells_fine(std::pow(std::pow(2, subdivision_order), ndim)) {
    NESOASSERT(ndim > 0, "ndim negative");
    NESOASSERT(dims.size() >= static_cast<std::size_t>(ndim),
               "vector of dims too small");
    NESOASSERT(origin.size() >= static_cast<std::size_t>(ndim),
               "vector of origin too small");
    for (int dimx = 0; dimx < ndim; dimx++) {
      NESOASSERT(dims[dimx] > 0, "Dim size is <= 0 in a direction.");
    }
    NESOASSERT(cell_width_coarse > 0.0, "Extent <= 0.0 passed");
    NESOASSERT(subdivision_order >= 0, "Negative subdivision order passed.");
    ncells_dim_fine = std::pow(2, subdivision_order);

    NESOASSERT(ncells_fine > 0, "Number of fine ncells does not make sense.");

    ncells_global = ncells_coarse * ncells_fine;

    // allocate space for the map stored on each shared memory region
    const MPI_Aint num_alloc =
        (comm_pair.rank_intra == 0) ? ncells_global * sizeof(int) : 0;
    MPICHK(MPI_Win_allocate_shared(num_alloc, sizeof(int), MPI_INFO_NULL,
                                   comm_pair.comm_intra,
                                   (void *)&this->map_base, &this->map_win))

    map_allocated = true;
    // map_base points to this ranks region - we want the base to the entire
    // allocated block in map.
    MPI_Aint win_size_tmp;
    int disp_unit_tmp;
    MPICHK(MPI_Win_shared_query(this->map_win, 0, &win_size_tmp, &disp_unit_tmp,
                                (void *)&this->map))
    NESOASSERT(static_cast<MPI_Aint>(ncells_global * sizeof(int)) ==
                   win_size_tmp,
               "Pointer to incorrect size.");
  };

  /**
   *  Free all data structures associated with this object. Must be called
   *  collectively on the communicator.
   */
  inline void free() {
    if (this->map_allocated) {
      MPICHK(MPI_Win_free(&this->map_win))
      this->map = NULL;
      this->map_base = NULL;
      this->map_allocated = false;
    }
    this->comm_pair.free();
    this->global_move_communication->free();
  }

  /**
   * Convert a global index represented by a tuple into a linear index.
   * tuple should be:
   * 1D: (coarse_x, fine_x)
   * 2D: (coarse_x, coarse_y, fine_x, fine_y)
   * 3D: (coarse_x, coarse_y, coarse_z, fine_x, fine_y, fine_z)
   *
   * @param index_tuple Index represented as a tuple.
   * @returns Linear index.
   */
  inline INT tuple_to_linear_global(INT *index_tuple) {
    INT index_coarse = tuple_to_linear_coarse(index_tuple);
    INT index_fine = tuple_to_linear_fine(&index_tuple[ndim]);
    INT index = index_coarse * ncells_fine + index_fine;
    return index;
  };

  /**
   * Convert a coarse mesh index represented by a tuple into a linear index.
   * tuple should be:
   * 1D: (coarse_x)
   * 2D: (coarse_x, coarse_y)
   * 3D: (coarse_x, coarse_y, coarse_z)
   *
   * @param index_tuple Index represented as a tuple.
   * @returns Linear index.
   */
  inline INT tuple_to_linear_coarse(INT *index_tuple) {
    INT index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= dims[dimx];
      index += index_tuple[dimx];
    }
    return index;
  };

  /**
   * Convert a fine mesh index represented by a tuple into a linear index.
   * tuple should be:
   * 1D: (fine_x)
   * 2D: (fine_x, fine_y)
   * 3D: (fine_x, fine_y, fine_z)
   *
   * @param index_tuple Index represented as a tuple.
   * @returns Linear index.
   */
  inline INT tuple_to_linear_fine(INT *index_tuple) {
    INT index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= ncells_dim_fine;
      index += index_tuple[dimx];
    }
    return index;
  };

  /**
   * Convert a global mesh linear index into a tuple index.
   * tuple should be:
   * 1D: (coarse_x, fine_x)
   * 2D: (coarse_x, coarse_y, fine_x, fine_y)
   * 3D: (coarse_x, coarse_y, coarse_z, fine_x, fine_y, fine_z)
   *
   * @param linear Linear index to convert.
   * @param index Output, index represented as a tuple.
   */
  inline void linear_to_tuple_global(INT linear, INT *index) {
    auto pq = std::div((long long)linear, (long long)ncells_fine);
    linear_to_tuple_coarse(pq.quot, index);
    linear_to_tuple_fine(pq.rem, index + ndim);
  };

  /**
   * Convert a coarse mesh linear index into a tuple index.
   * tuple should be:
   * 1D: (coarse_x,)
   * 2D: (coarse_x, coarse_y)
   * 3D: (coarse_x, coarse_y, coarse_z,)
   *
   * @param linear Linear index to convert.
   * @param index Output, index represented as a tuple.
   */
  inline void linear_to_tuple_coarse(INT linear, INT *index) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div((long long)linear, (long long)dims[dimx]);
      index[dimx] = pq.rem;
      linear = pq.quot;
    }
  };

  /**
   * Convert a fine mesh linear index into a tuple index.
   * tuple should be:
   * 1D: (fine_x,)
   * 2D: (fine_x, fine_y)
   * 3D: (fine_x, fine_y, fine_z)
   *
   * @param linear Linear index to convert.
   * @param index Output, index represented as a tuple.
   */
  inline void linear_to_tuple_fine(INT linear, INT *index) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div((long long)linear, (long long)ncells_dim_fine);
      index[dimx] = pq.rem;
      linear = pq.quot;
    }
  };

  /**
   * Mark the start of an epoch where ranks claim global cells. Collective on
   * communicator.
   */
  inline void claim_initialise() {
    NESOASSERT(claim_stack.empty(), "Claim stack is not empty.");

    // initialise the memory in the shared memory window
    MPICHK(MPI_Win_fence(0, this->map_win))
    INT start, end;
    get_decomp_1d((INT)this->comm_pair.size_intra, ncells_global,
                  (INT)this->comm_pair.rank_intra, &start, &end);
    for (INT cellx = start; cellx < end; cellx++) {
      map[cellx] = mask;
    }
    MPICHK(MPI_Win_fence(0, this->map_win))

    MPICHK(MPI_Barrier(this->comm_pair.comm_intra))
    MPICHK(MPI_Win_fence(0, this->map_win))
  }

  /**
   * Claim a cell in the MeshHierarchy with a certain weight. The rank that
   * claims with the highest weight owns the cell. In the case of weight
   * contention the highest rank is given the cell. Weights should be an
   * integer in the range '[1, int_max-1]'. Claim with the highest weight will
   * be assigned the cell.
   *
   * @param index Global linear index to cell to claim.
   * @param weight Weight to claim the cell with.
   */
  inline void claim_cell(const INT index, int weight) {
    NESOASSERT((weight != mask) && (weight > 0) && (weight < (-1 * (mask + 1))),
               "Invalid weight");
    weight += mask;
    std::pair<INT, int> claim{index, weight};
    claim_stack.push(claim);

    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, map_win);
    int result;
    MPICHK(MPI_Get_accumulate(&weight, 1, MPI_INT, &result, 1, MPI_INT, 0,
                              (MPI_Aint)index, 1, MPI_INT, MPI_MAX, map_win));

    MPI_Win_unlock(0, map_win);
  }

  /**
   * Mark the end of an epoch where ranks claim global cells. Collective on
   * communicator.
   */
  inline void claim_finalise() {

    MPICHK(MPI_Win_fence(0, this->map_win));
    MPICHK(MPI_Barrier(this->comm_pair.comm_intra));

    // reduce accross all shared regions (inter node)
    all_reduce_max_map();

    NESOASSERT(claim_stack_binned.empty(), "Claim stack is not empty.");

    while (!claim_stack.empty()) {
      auto claim = claim_stack.top();

      // if the weight of this claim is in the cell then this rank is one of
      // the ranks with the maximum weight.
      if (claim.second >= map[claim.first]) {
        claim_stack_binned.push(claim);
      }
      claim_stack.pop();
    }

    MPICHK(MPI_Barrier(this->comm_pair.comm_intra));

    // for each rank with the maximum claimed weight reduce the ranks
    const int rank = comm_pair.rank_parent;

    MPICHK(MPI_Win_fence(0, this->map_win));

    while (!claim_stack_binned.empty()) {
      auto claim = claim_stack_binned.top();
      INT index = claim.first;
      int result;

      MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, map_win);
      MPICHK(MPI_Get_accumulate(&rank, 1, MPI_INT, &result, 1, MPI_INT, 0,
                                (MPI_Aint)index, 1, MPI_INT, MPI_MAX, map_win));
      MPI_Win_unlock(0, map_win);
      claim_stack_binned.pop();
    }

    MPICHK(MPI_Win_fence(0, this->map_win));

    MPICHK(MPI_Barrier(this->comm_pair.comm_intra));
    // reduce accross all shared regions (inter node)
    all_reduce_max_map();

    // this->map should now contain either values of "mask" for cells that were
    // not claimed and the rank of claimed cells.
    map_created = true;
  }

  /**
   *  Get the owning MPI rank for a linear cell index.
   *
   *  @param index Global linear index of cell to query the owning rank of.
   *  @returns Owning MPI rank.
   */
  inline int get_owner(INT index) {
    NESOASSERT(map_created, "map is not created");
    return map[index];
  };

  /**
   *  Get the owning MPI ranks for n indices in global tuple form.
   *
   *  @param nqueries Number of cells to query.
   *  @param indices Array containing `nqueries` global index tuples.
   *  @param ranks Array owning ranks will be stored in.
   */
  inline void get_owners(const int nqueries, INT *indices, int *ranks) {
    for (int qx = 0; qx < nqueries; qx++) {
      const INT linear_index =
          this->tuple_to_linear_global(indices + (qx * this->ndim * 2));
      const int rank = get_owner(linear_index);
      ranks[qx] = rank;
      if (Debug::enabled(Debug::MOVEMENT_LEVEL)) {
        nprint("MeshHierarchy::get_owners:", "qx:", qx,
               "linear_index:", linear_index, "rank:", rank);
      }
    }
  };
};

/**
 *  A struct to aid binning points into MeshHierarchy cells. This type is
 *  device copyable such that points can be easilly binned into cells on the
 *  device.
 */
struct MeshHierarchyDeviceMapper {
  /// Number of spatial dimensions.
  INT ndim;
  /// ndim sized array holding the mesh origin.
  REAL *origin;
  /// ndim sized array holding the number of coarse cells in each direction.
  INT *dims;
  /// The inverse of the coarse cell width.
  REAL inverse_cell_width_coarse;
  /// The inverse of the fine cell width.
  REAL inverse_cell_width_fine;
  /// The coarse cell width.
  REAL cell_width_coarse;
  /// The fine cell width.
  REAL cell_width_fine;
  /// The number of fine cells in each dimension per dimension.
  INT ncells_dim_fine;
  /// The total number of fine cells per coarse cell.
  INT ncells_fine;

  /**
   * For an input point find the MeshHierarchy cell that contains the point in
   * tuple form. Output tuple should be: 1D: (coarse_x, fine_x) 2D: (coarse_x,
   * coarse_y, fine_x, fine_y) 3D: (coarse_x, coarse_y, coarse_z, fine_x,
   * fine_y, fine_z)
   *
   * @param[in] position Input position.
   * @param[out] cell Ouput, index represented as a tuple.
   */
  inline void map_to_tuple(const REAL *position, INT *cell) const {
    for (int dimx = 0; dimx < ndim; dimx++) {
      // position relative to the mesh origin
      const REAL pos = position[dimx] - origin[dimx];
      const REAL tol = 1.0e-10;

      // coarse grid index
      INT cell_coarse = ((REAL)pos * inverse_cell_width_coarse);
      // bounds check the cell at the upper extent
      if (cell_coarse >= dims[dimx]) {
        // if the particle is within a given tolerance assume the
        // out of bounds is a floating point issue.
        if ((ABS(pos - dims[dimx] * cell_width_coarse) / ABS(pos)) <= tol) {
          cell_coarse = dims[dimx] - 1;
          cell[dimx] = cell_coarse;
        } else {
          cell_coarse = 0;
          cell[dimx] = -2;
        }
      } else {
        cell[dimx] = cell_coarse;
      }

      // use the coarse cell index to offset the origin and compute
      // the fine cell index
      const REAL pos_fine = pos - cell_coarse * cell_width_coarse;
      INT cell_fine = ((REAL)pos_fine * inverse_cell_width_fine);

      if (cell_fine >= ncells_dim_fine) {
        if ((ABS(pos_fine - ncells_dim_fine * cell_width_fine) /
             ABS(pos_fine)) <= tol) {
          cell_fine = ncells_dim_fine - 1;
          cell[dimx + ndim] = cell_fine;
        } else {
          cell[dimx + ndim] = -2;
        }
      } else {
        cell[dimx + ndim] = cell_fine;
      }
    }
  }

  /**
   * For an input MeshHierarchy tuple output the tuple which would index into a
   * standard Cartesian mesh of the same structure.
   * Input tuple should be:
   * 1D: (coarse_x, fine_x)
   * 2D: (coarse_x, coarse_y, fine_x, fine_y)
   * 3D: (coarse_x, coarse_y, coarse_z, fine_x, fine_y, fine_z)
   *
   * @param[in] mh_tuple Input tuple in global MeshHierarchy form.
   * @param[out] cart_tuple N dimensional tuple which indexes into the Cartesian
   * mesh view of the MeshHierarchy.
   */
  inline void map_tuple_to_cart_tuple(const INT *mh_tuple,
                                      INT *cart_tuple) const {
    for (int dimx = 0; dimx < ndim; dimx++) {
      cart_tuple[dimx] = mh_tuple[ndim + dimx] + mh_tuple[dimx];
    }
  }

  /**
   * Map position into the cartesian grid. Do not attempt to trucate the
   * position into the mesh.
   *
   * @param[in] position Input position to map into the Cartesian view of the
   * MeshHierarchy.
   * @param[in, out] cell N dimensional tuple that maps into the Cartesian view
   * of the MeshHierarchy.
   */
  inline void map_to_cart_tuple_no_trunc(const REAL *position,
                                         INT *cell) const {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const REAL origin_dim = origin[dimx];
      const REAL pos = position[dimx];
      if (pos < origin_dim) {
        const REAL offset = origin_dim - pos;
        const REAL real_cell = offset * inverse_cell_width_fine;
        const INT int_cell = static_cast<INT>(real_cell) + 1;
        const INT relative_cell = -int_cell;
        cell[dimx] = relative_cell;
      } else {
        const REAL real_cell = (pos - origin_dim) * inverse_cell_width_fine;
        // rounds towards 0
        const INT int_cell = static_cast<INT>(real_cell);
        cell[dimx] = int_cell;
      }
    }
  }

  /**
   * Convert a Cartesian tuple index into a MeshHierarchy global tuple index.
   *
   * @param[in] cart_tuple Cartesian tuple to convert.
   * @param[in, out] mh_tuple MeshHierarchy global tuple representation of input
   * tuple.
   */
  inline void cart_tuple_to_tuple(const INT *cart_tuple, INT *mh_tuple) const {
    for (int dimx = 0; dimx < ndim; dimx++) {
      const INT cell_coarse = cart_tuple[dimx] / ncells_dim_fine;
      const INT cell_fine = cart_tuple[dimx] % ncells_dim_fine;
      mh_tuple[dimx] = cell_coarse;
      mh_tuple[ndim + dimx] = cell_fine;
    }
  }

  /**
   * Convert a global index represented by a tuple into a linear index.
   * tuple should be:
   * 1D: (coarse_x, fine_x)
   * 2D: (coarse_x, coarse_y, fine_x, fine_y)
   * 3D: (coarse_x, coarse_y, coarse_z, fine_x, fine_y, fine_z)
   *
   * @param index_tuple Index represented as a tuple.
   * @returns Linear index.
   */
  inline INT tuple_to_linear_global(INT *index_tuple) const {
    INT index_coarse = tuple_to_linear_coarse(index_tuple);
    INT index_fine = tuple_to_linear_fine(&index_tuple[ndim]);
    INT index = index_coarse * ncells_fine + index_fine;
    return index;
  };

  /**
   * Convert a coarse mesh index represented by a tuple into a linear index.
   * tuple should be:
   * 1D: (coarse_x)
   * 2D: (coarse_x, coarse_y)
   * 3D: (coarse_x, coarse_y, coarse_z)
   *
   * @param index_tuple Index represented as a tuple.
   * @returns Linear index.
   */
  inline INT tuple_to_linear_coarse(INT *index_tuple) const {
    INT index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= dims[dimx];
      index += index_tuple[dimx];
    }
    return index;
  };

  /**
   * Convert a fine mesh index represented by a tuple into a linear index.
   * tuple should be:
   * 1D: (fine_x)
   * 2D: (fine_x, fine_y)
   * 3D: (fine_x, fine_y, fine_z)
   *
   * @param index_tuple Index represented as a tuple.
   * @returns Linear index.
   */
  inline INT tuple_to_linear_fine(INT *index_tuple) const {
    INT index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= ncells_dim_fine;
      index += index_tuple[dimx];
    }
    return index;
  }
};

/**
 * A class to provide methods to bin points into MeshHierarchy cells and
 * transform between representations of MeshHierarchy indexing.
 */
class MeshHierarchyMapper {
private:
  std::shared_ptr<BufferDeviceHost<REAL>> dh_origin;
  std::shared_ptr<BufferDeviceHost<INT>> dh_dims;
  std::shared_ptr<MeshHierarchy> mesh_hierarchy;

  inline MeshHierarchyDeviceMapper get_generic_mapper() {

    MeshHierarchyDeviceMapper mapper;
    mapper.ndim = this->mesh_hierarchy->ndim;
    mapper.inverse_cell_width_coarse =
        this->mesh_hierarchy->inverse_cell_width_coarse;
    mapper.inverse_cell_width_fine =
        this->mesh_hierarchy->inverse_cell_width_fine;
    mapper.cell_width_coarse = this->mesh_hierarchy->cell_width_coarse;
    mapper.cell_width_fine = this->mesh_hierarchy->cell_width_fine;
    mapper.ncells_dim_fine = this->mesh_hierarchy->ncells_dim_fine;
    mapper.ncells_fine = this->mesh_hierarchy->ncells_fine;
    return mapper;
  }

public:
  /**
   * Create new instance of helper mapper class.
   *
   * @param[in] sycl_target The compute device to provide device copyable mapper
   * structs for.
   * @param[in] mesh_hierarchy The MeshHierarchy for which to create a mapper
   * class for.
   */
  MeshHierarchyMapper(SYCLTargetSharedPtr sycl_target,
                      std::shared_ptr<MeshHierarchy> mesh_hierarchy)
      : mesh_hierarchy(mesh_hierarchy) {

    static_assert(
        std::is_trivially_copyable_v<MeshHierarchyDeviceMapper> == true,
        "MeshHierarchyDeviceMapper is not trivially copyable to device");

    const int ndim = mesh_hierarchy->ndim;
    std::vector<INT> dims_tmp(ndim);
    std::vector<REAL> origin_tmp(ndim);
    for (int dimx = 0; dimx < ndim; dimx++) {
      dims_tmp[dimx] = mesh_hierarchy->dims[dimx];
      origin_tmp[dimx] = mesh_hierarchy->origin[dimx];
    }
    this->dh_origin =
        std::make_shared<BufferDeviceHost<REAL>>(sycl_target, origin_tmp);
    this->dh_dims =
        std::make_shared<BufferDeviceHost<INT>>(sycl_target, dims_tmp);
  }

  /**
   * Get a device copyable struct with device callable methods for interfacing
   * with a MeshHierarchy.
   *
   * @returns New mapper struct.
   */
  inline MeshHierarchyDeviceMapper get_device_mapper() {
    MeshHierarchyDeviceMapper mapper = this->get_generic_mapper();
    mapper.origin = this->dh_origin->d_buffer.ptr;
    mapper.dims = this->dh_dims->d_buffer.ptr;
    return mapper;
  }

  /**
   * Get a host struct with methods to interface with a MeshHierarchy instance.
   *
   * @returns New mapper struct.
   */
  inline MeshHierarchyDeviceMapper get_host_mapper() {
    MeshHierarchyDeviceMapper mapper = this->get_generic_mapper();
    mapper.origin = this->dh_origin->h_buffer.ptr;
    mapper.dims = this->dh_dims->h_buffer.ptr;
    return mapper;
  }
};

} // namespace NESO::Particles

#endif
