#ifndef _NESO_PARTICLES_HIERARCHY
#define _NESO_PARTICLES_HIERARCHY
#include "communication.hpp"
#include "compute_target.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mpi.h>
#include <stack>
#include <utility>
#include <vector>

namespace NESO::Particles {

const int mask = std::numeric_limits<int>::min();

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
  MPI_Comm comm;
  CommPair comm_pair;
  int ndim;
  std::vector<int> dims;
  std::vector<double> origin;
  int subdivision_order;

  double cell_width_coarse;
  double cell_width_fine;
  double inverse_cell_width_coarse;
  double inverse_cell_width_fine;

  INT ncells_coarse;
  INT ncells_fine;
  INT ncells_dim_fine;
  INT ncells_global;

  MeshHierarchy(){};
  MeshHierarchy(MPI_Comm comm, const int ndim, std::vector<int> dims,
                std::vector<double> origin, const double extent = 1.0,
                const int subdivision_order = 1)
      : comm(comm), comm_pair(comm), ndim(ndim), dims(dims), origin(origin),
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
    ncells_dim_fine = std::pow(2, subdivision_order);

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
    NESOASSERT(ncells_global * sizeof(int) == win_size_tmp,
               "Pointer to incorrect size.");
  };
  inline void free() {
    if (map_allocated) {
      MPICHK(MPI_Win_free(&this->map_win))
      this->map = NULL;
      this->map_base = NULL;
      map_allocated = false;
    }
    comm_pair.free();
  }

  /*
   * tuple should be:
   * 1D: (coarse_x, fine_x)
   * 2D: (coarse_x, coarse_y, fine_x, fine_y)
   * 3D: (coarse_x, coarse_y, coarse_z, fine_x, fine_y, fine_z)
   */
  inline INT tuple_to_linear_global(INT *index_tuple) {
    INT index_coarse = tuple_to_linear_coarse(index_tuple);
    INT index_fine = tuple_to_linear_fine(&index_tuple[ndim]);
    INT index = index_coarse * ncells_fine + index_fine;
    return index;
  };
  inline INT tuple_to_linear_coarse(INT *index_tuple) {
    INT index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= dims[dimx];
      index += index_tuple[dimx];
    }
    return index;
  };
  inline INT tuple_to_linear_fine(INT *index_tuple) {
    INT index = index_tuple[ndim - 1];
    for (int dimx = ndim - 2; dimx >= 0; dimx--) {
      index *= ncells_dim_fine;
      index += index_tuple[dimx];
    }
    return index;
  };
  inline void linear_to_tuple_global(INT linear, INT *index) {
    auto pq = std::div((long long)linear, (long long)ncells_fine);
    linear_to_tuple_coarse(pq.quot, index);
    linear_to_tuple_fine(pq.rem, index + ndim);
  };
  inline void linear_to_tuple_coarse(INT linear, INT *index) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div((long long)linear, (long long)dims[dimx]);
      index[dimx] = pq.rem;
      linear = pq.quot;
    }
  };
  inline void linear_to_tuple_fine(INT linear, INT *index) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      auto pq = std::div((long long)linear, (long long)ncells_dim_fine);
      index[dimx] = pq.rem;
      linear = pq.quot;
    }
  };

  /*
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

  /*
   * Claim a cell in the MeshHierarchy with a certain weight. The rank that
   * claims with the highest weight owns the cell. In the case of weight
   * contention the highest rank is given the cell.
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

  /*
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

  /*
   *  Get the owning MPI rank for a linear cell index.
   */
  inline int get_owner(INT index) {
    NESOASSERT(map_created, "map is not created");
    return map[index];
  };

  /*
   *  Get the owning MPI ranks for n indicies in global tuple form.
   */
  inline void get_owners(const int nqueries, INT *indices, int *ranks) {
    for (int qx = 0; qx < nqueries; qx++) {
      const INT linear_index =
          this->tuple_to_linear_global(indices + (qx * this->ndim * 2));
      const int rank = get_owner(linear_index);
      ranks[qx] = rank;
    }
  };
};

} // namespace NESO::Particles

#endif
