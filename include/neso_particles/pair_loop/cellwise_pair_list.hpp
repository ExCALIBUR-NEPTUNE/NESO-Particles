#ifndef __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_HPP_
#define __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../device_buffers.hpp"

#include <map>
#include <vector>

namespace NESO::Particles {

/**
 * Device type for CellwisePairList.
 */
struct CellwisePairListDevice {
  int num_waves{0};
  int cell_count{0};

  // These pointer types should be accessed through the methods not directly.
  int *const *const *d_pair_list{nullptr};
  int const *d_pair_counts{nullptr};
  INT const *d_pair_counts_es{nullptr};
  int const *h_pair_counts{nullptr};

  int max_index{0};
  int max_pair_count{0};
  INT pair_count{0};

  /**
   * @param cell Cell to retrieve number of waves for.
   * @returns The number of waves for the given cell.
   */
  inline int get_num_waves(const int cell) const { return this->num_waves; }

  /**
   * @param wave Wave to retreive number of pairs for.
   * @param cell Cell to retrieve number of pairs for.
   * @returns The number of pairs in the given wave and given cell.
   */
  inline int get_num_pairs(const int wave, const int cell) const {
    return this->d_pair_counts[wave * this->cell_count + cell];
  }

  /**
   * @param wave Wave to retreive number of pairs for.
   * @param cell Cell to retrieve number of pairs for.
   * @returns The number of pairs in the given wave and given cell.
   */
  inline int get_num_pairs_host(const int wave, const int cell) const {
    return this->h_pair_counts[wave * this->cell_count + cell];
  }

  /**
   *
   * @param wave Wave to retreive offset for.
   * @param cell Cell to retreive offset for.
   * @returns The offset to add to the pair index to compute the pair index in
   * this pair list.
   */
  inline int get_pair_index_offset(const int wave, const int cell) const {
    return this->d_pair_counts_es[wave * this->cell_count + cell];
  }

  /**
   * @param wave Wave to retrieve particle index for.
   * @param cell Cell to retrieve particle index for.
   * @param pair_index Pair index to retrieve particle index for.
   * @returns First particle index in pair.
   */
  inline int get_particle_index_i(const int wave, const int cell,
                                  const int pair_index) const {
    return this->d_pair_list[cell][0][pair_index];
  }

  /**
   * @param wave Wave to retrieve particle index for.
   * @param cell Cell to retrieve particle index for.
   * @param pair_index Pair index to retrieve particle index for.
   * @returns Second particle index in pair.
   */
  inline int get_particle_index_j(const int wave, const int cell,
                                  const int pair_index) const {
    return this->d_pair_list[cell][1][pair_index];
  }
};

/**
 * Type to hold a list of pairs of particles that should collide in each cell.
 */
class CellwisePairList {
protected:
  // The number of waves. Pairs within a wave are independent.
  int num_waves{0};
  // The pair list itself.
  std::shared_ptr<CellDat<int>> d_pair_list;
  // The number of pairs in each cell (device)
  std::shared_ptr<BufferDevice<int>> d_pair_counts;
  // Exclusive scan of the number of pairs in each cell.
  std::shared_ptr<BufferDevice<INT>> d_pair_counts_es;
  // This number of pairs in each cell (host)
  std::vector<int> h_pair_counts;
  // The max index in the pair list
  int max_index{-1};
  // The max size of the pair lists across all cells.
  int max_pair_count{-1};
  // The number of pairs.
  INT pair_count{-1};

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

} // namespace NESO::Particles

#endif
