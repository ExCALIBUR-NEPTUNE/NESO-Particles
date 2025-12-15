#ifndef __NESO_PARTICLES_PAIR_LOOP_CELLWISE_PAIR_LIST_BLOCK_HPP_
#define __NESO_PARTICLES_PAIR_LOOP_CELLWISE_PAIR_LIST_BLOCK_HPP_

#include "../compute_target.hpp"
#include "../device_buffers.hpp"

#include <map>
#include <vector>

namespace NESO::Particles {

/**
 * Device type for CellwisePairListBlock.
 */
struct CellwisePairListBlockDevice {

  INT pair_count{0};
  int block_size{0};
  int cell_count{0};
  int max_wave_count{0};
  int *d_wave_counts{nullptr};
  int const *d_pair_counts{nullptr};
  INT const *d_pair_counts_es{nullptr};
  int const *d_pair_list{nullptr};

  inline int get_num_waves(const int cell, const int block) const {
    return this->d_wave_counts[block * this->cell_count + cell];
  }

  inline int get_num_pairs(const int cell) const {
    return this->d_pair_counts[cell];
  }

  inline int get_pair_linear_index(const int cell, const int pair_index) const {
    return this->d_pair_counts_es[cell] + pair_index;
  }

  inline int get_pair_index_i(const int cell, const int pair_index) const {
    const int offset = 0;
    const int index = this->get_pair_linear_index(cell, pair_index);
    return this->d_pair_list[offset + index];
  }

  inline int get_pair_index_j(const int cell, const int pair_index) const {
    const int offset = this->pair_count;
    const int index = this->get_pair_linear_index(cell, pair_index);
    return this->d_pair_list[offset + index];
  }

  inline int get_pair_wave(const int cell, const int pair_index) const {
    const int offset = this->pair_count * 2;
    const int index = this->get_pair_linear_index(cell, pair_index);
    return this->d_pair_list[offset + index];
  }
};

/**
 * Abstract interface class for classes which generate
 * CellwisePairListBlockDevice based pair loops.
 */
struct CellwisePairListBlockInterface {
  virtual ~CellwisePairListBlockInterface() = default;

  /**
   * @returns The device copyable representation of the pair list.
   */
  virtual CellwisePairListBlockDevice get_pair_list() = 0;

  /**
   *
   * @param sycl_target Compute target for the pair list.
   * @returns Host representation of the pair list.
   */
  std::map<int,
           std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>>
  get_host_pair_list(SYCLTargetSharedPtr sycl_target);

  /**
   * @param sycl_target Compute target for the pair list.
   * @returns True if no conflicts were detected or the pair list was otherwise
   * malformed.
   */
  bool validate_pair_list(SYCLTargetSharedPtr sycl_target);

  /**
   * @param[in] sycl_target Compute target for the pair list.
   * @param[in, out] occupancy_counts Vector of size block_size + 1. The entry
   * at index i denotes the number of blocks that execute i pairs.
   */
  void get_wave_occupancy_counts(SYCLTargetSharedPtr sycl_target,
                                 std::vector<int> &occupancy_counts);
};

/**
 * @param occupancy_counts Vector of occupancies of size block_size + 1. See
 * get_wave_occupancy_counts.
 * @returns Average proportion of wave which is active.
 */
REAL get_mean_wave_occupancy(std::vector<int> &occupancy_counts);

/**
 * Reduce wave occupancy counts onto rank zero.
 *
 * @param[in] SYCLTargetSharedPtr Compute target to reduce over.
 * @param[in] local_occupancy_counts Local contributions to occupancy counts.
 * @param[in, out] global_occupancy_counts Global occupancy counts.
 */
void get_global_wave_occupancy_counts(
    SYCLTargetSharedPtr sycl_target, std::vector<int> &local_occupancy_counts,
    std::vector<int> &global_occupancy_counts);

using CellwisePairListBlockInterfaceSharedPtr =
    std::shared_ptr<CellwisePairListBlockInterface>;

} // namespace NESO::Particles

#endif
