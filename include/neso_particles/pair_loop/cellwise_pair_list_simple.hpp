#ifndef __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_SIMPLE_HPP_
#define __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_SIMPLE_HPP_

#include "cellwise_pair_list.hpp"

namespace NESO::Particles {

/**
 * Type to hold a list of pairs of particles that should collide in each cell.
 */
class CellwisePairListSimple : public CellwisePairList {
protected:
  // The number of waves. Pairs within a wave are independent.
  std::vector<int> h_wave_count;
  // The number of waves. Pairs within a wave are independent.
  std::shared_ptr<BufferDevice<int>> d_wave_count;
  // The offsets for each cell to the start of the wave in the pair list.
  std::shared_ptr<BufferDevice<int>> d_wave_offsets;
  std::vector<int> h_wave_offsets;
  // The pair list itself.
  std::shared_ptr<CellDat<int>> d_pair_list;
  // The number of pairs in each cell (device)
  std::shared_ptr<BufferDevice<int>> d_pair_counts;
  // Exclusive scan of the number of pairs in each cell.
  std::shared_ptr<BufferDevice<INT>> d_pair_counts_es;
  std::vector<INT> h_pair_counts_es;
  // This number of pairs in each cell (host)
  std::vector<int> h_pair_counts;
  // The max size of the pair lists across all cells.
  int max_pair_count{-1};
  // The max wave count of any cell
  int max_wave_count{-1};
  // The number of pairs.
  INT pair_count{-1};

  int mode{0};

public:
  /// Compute device holding pairs.
  SYCLTargetSharedPtr sycl_target;
  /// Number of cells to hold pairs for.
  int cell_count{0};

  /// Disable (implicit) copies.
  CellwisePairListSimple(const CellwisePairListSimple &st) = delete;
  /// Disable (implicit) copies.
  CellwisePairListSimple &operator=(CellwisePairListSimple const &a) = delete;
  ~CellwisePairListSimple() = default;

  /**
   * @param sycl_target Compute device for pairs.
   * @param cell_count Number of cells to hold pairs for.
   */
  CellwisePairListSimple(SYCLTargetSharedPtr sycl_target, const int cell_count);

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
   * Set the cell list from a host description of waves of pairs.
   *
   * @param pair_list Host description of pairs.
   */
  void set(CellwisePairListHostSharedPtr pair_list);

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
  virtual CellwisePairListDevice get_pair_list() override;

  /**
   * Get a description of the pair list accessible on the host.
   *
   * @returns Container of pairs for each cell.
   */
  virtual CellwisePairListHostMap get_host_pair_list() override;
};

using CellwisePairListSimpleSharedPtr = std::shared_ptr<CellwisePairListSimple>;

} // namespace NESO::Particles

#endif
