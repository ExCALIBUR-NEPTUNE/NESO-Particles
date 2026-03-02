#ifndef __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_HPP_
#define __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../device_buffers.hpp"
#include "cellwise_pair_list_host.hpp"

#include <map>
#include <vector>

namespace NESO::Particles {

/**
 * Device type for CellwisePairList.
 */
struct CellwisePairListDevice {
  int const *h_wave_count{nullptr};
  int const *d_wave_count{nullptr};
  int cell_count{0};

  // These pointer types should be accessed through the methods not directly.
  int const *d_wave_offsets{nullptr};
  int *const *const *d_pair_list{nullptr};
  int const *d_pair_counts{nullptr};
  INT const *d_pair_counts_es{nullptr};
  int const *h_pair_counts{nullptr};

  // max pair count of any wave
  int max_pair_count{0};
  int max_wave_count{0};
  INT pair_count{0};

  /**
   * @param cell Cell to retrieve number of waves for.
   * @returns The number of waves for the given cell.
   */
  inline int get_num_waves(const int cell) const {
    return this->d_wave_count[cell];
  }

  /**
   * @param wave Wave to retrieve number of pairs for.
   * @param cell Cell to retrieve number of pairs for.
   * @returns The number of pairs in the given wave and given cell.
   */
  inline int get_num_pairs(const int wave, const int cell) const {
    return this->d_pair_counts[wave * this->cell_count + cell];
  }

  /**
   * @param wave Wave to retrieve number of pairs for.
   * @param cell Cell to retrieve number of pairs for.
   * @returns The number of pairs in the given wave and given cell.
   */
  inline int get_num_pairs_host(const int wave, const int cell) const {
    return this->h_pair_counts[wave * this->cell_count + cell];
  }

  /**
   * @param wave Wave to retrieve offset for.
   * @param cell Cell to retrieve offset for.
   * @param pair_index Pair index to retrieve index for.
   * @returns The linear index of the pair in this pair list.
   */
  inline int get_pair_linear_index(const int wave, const int cell,
                                   const int pair_index) const {
    return this->d_pair_counts_es[wave * this->cell_count + cell] + pair_index;
  }

  /**
   * @param wave Wave to retrieve particle index for.
   * @param cell Cell to retrieve particle index for.
   * @param pair_index Pair index to retrieve particle index for.
   * @returns First particle index in pair.
   */
  inline int get_particle_index_i(const int wave, const int cell,
                                  const int pair_index) const {
    const int offset = this->d_wave_offsets[wave * this->cell_count + cell];
    return this->d_pair_list[cell][0][offset + pair_index];
  }

  /**
   * @param wave Wave to retrieve particle index for.
   * @param cell Cell to retrieve particle index for.
   * @param pair_index Pair index to retrieve particle index for.
   * @returns Second particle index in pair.
   */
  inline int get_particle_index_j(const int wave, const int cell,
                                  const int pair_index) const {

    const int offset = this->d_wave_offsets[wave * this->cell_count + cell];
    return this->d_pair_list[cell][1][offset + pair_index];
  }
};

/**
 * Abstract type to hold a list of pairs of particles that should collide in
 * each cell.
 */
class CellwisePairList {

public:
  virtual ~CellwisePairList() = default;

  /**
   * Get a description of the pair list accessible on the device.
   *
   * @returns PairListDevice describing all the particle pairs. The returned
   * object must have a lifetime equal or shorter than this host instance.
   */
  virtual CellwisePairListDevice get_pair_list() = 0;

  /**
   * Get a description of the pair list accessible on the host.
   *
   * @returns Container of pairs for each cell.
   */
  virtual CellwisePairListHostMap get_host_pair_list() = 0;
};

using CellwisePairListSharedPtr = std::shared_ptr<CellwisePairList>;

} // namespace NESO::Particles

#endif
