#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_PAIR_LIST_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_PAIR_LIST_HPP_

#include "../../compute_target.hpp"
#include "../../containers/cell_dat.hpp"
#include "../../device_buffers.hpp"

#include <map>
#include <vector>

namespace NESO::Particles::DSMC {

/**
 * Device type for CellwisePairList.
 */
struct CellwisePairListDevice {
  int cell_count;
  int ***d_pair_list;
  int *d_pair_counts;
  int *h_pair_counts;
  int max_index;
};

/**
 * Type to hold a list of pairs of particles that should collide in each cell.
 */
class CellwisePairList {
protected:
  // The pair list itself.
  std::shared_ptr<CellDat<int>> d_pair_list;
  // The number of pairs in each cell (device)
  std::shared_ptr<BufferDevice<int>> d_pair_counts;
  // This number of pairs in each cell (host)
  std::vector<int> h_pair_counts;
  // The max index in the pair list
  int max_index{-1};

public:
  /// Compute device holding pairs.
  SYCLTargetSharedPtr sycl_target;
  /// Number of cells to hold pairs for.
  int cell_count{0};

  /// Disable (implicit) copies.
  CellwisePairList(const CellwisePairList &st) = delete;
  /// Disable (implicit) copies.
  CellwisePairList &operator=(CellwisePairList const &a) = delete;
  ~CellwisePairList() = default;

  /**
   * @param sycl_target Compute device for pairs.
   * @param cell_count Number of cells to hold pairs for.
   */
  CellwisePairList(SYCLTargetSharedPtr sycl_target, const int cell_count);

  /**
   * Host callable utility method to push pairs (i,j) onto the cell list for
   * cells c.
   *
   * @param c Cells to push pair (i,j) onto the cell list for.
   * @param i First particles of the pair.
   * @param j Second particles of the pair.
   */
  void push_back(const std::vector<int> &c, const std::vector<int> &i,
                 const std::vector<int> &j);

  /**
   * Empty the pair list.
   */
  void clear();

  /**
   * Get a description of the pair list accessible on the device.
   *
   * @returns PairListDevice describing all the particle pairs. The returned
   * object must have a lifetime equal or shorter than this host instance.
   */
  CellwisePairListDevice get();

  /**
   * Get a description of the pair list accessible on the host.
   *
   * @returns Container of pairs for each cell.
   */
  std::map<int, std::pair<std::vector<int>, std::vector<int>>> host_get();
};

using CellwisePairListSharedPtr = std::shared_ptr<CellwisePairList>;

} // namespace NESO::Particles::DSMC

#endif
