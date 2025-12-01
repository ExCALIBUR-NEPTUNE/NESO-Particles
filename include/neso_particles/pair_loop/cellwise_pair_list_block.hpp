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
  int *d_wave_counts{nullptr};
  int const *d_pair_counts{nullptr};
  INT const *d_pair_counts_es{nullptr};
  int const *d_pair_list{nullptr};

  inline int get_wave_count(const int cell, const int block) {
    return this->d_wave_counts[block * this->cell_count + cell];
  }

  inline int get_num_pairs(const int cell) const {
    return this->d_pair_counts[cell];
  }

  inline int get_pair_linear_index(const int cell, const int pair_index) const {
    return this->d_pair_counts_es[cell] + pair_index;
  }

  inline int get_particle_index_i(const int cell, const int pair_index) const {
    const int offset = 0;
    const int index = this->get_pair_linear_index(cell, pair_index);
    return this->d_pair_list[offset + index];
  }

  inline int get_particle_index_j(const int cell, const int pair_index) const {
    const int offset = this->pair_count;
    const int index = this->get_pair_linear_index(cell, pair_index);
    return this->d_pair_list[offset + index];
  }

  inline int get_particle_wave(const int cell, const int pair_index) const {
    const int offset = this->pair_count * 2;
    const int index = this->get_pair_linear_index(cell, pair_index);
    return this->d_pair_list[offset + index];
  }
};

struct CellwisePairListBlockInterface {
  virtual CellwisePairListBlockDevice get_pair_list() = 0;

  std::map<int,
           std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>>
  get_host_pair_list(SYCLTargetSharedPtr sycl_target);
};

} // namespace NESO::Particles

#endif
