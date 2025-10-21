#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_PAIR_LIST_WAVES_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_PAIR_LIST_WAVES_HPP_

#include "cellwise_pair_list.hpp"
#include <memory>

namespace NESO::Particles::DSMC {

/**
 * Device type for CellwisePairListWaves.
 */
struct CellwisePairListWavesDevice {};

/**
 * TODO
 */
class CellwisePairListWaves {
protected:
  SYCLTargetSharedPtr sycl_target;
  std::size_t num_lists{0};

  std::vector<CellwisePairListDevice> h_pair_lists_device;
  std::shared_ptr<BufferDevice<CellwisePairListDevice>> d_pair_lists_device;
  int cell_count{0};

public:
  /// The original pair lists from which to compute waves.
  std::vector<std::shared_ptr<CellwisePairList>> cellwise_pair_lists;

  ~CellwisePairListWaves() = default;
  CellwisePairListWaves() = default;

  /**
   * TODO
   *
   * The passed pair lists must be completely disjoint. i.e. If an index i
   * appears in a cell list then i appears in no other passed cell list.
   *
   */
  CellwisePairListWaves(
      SYCLTargetSharedPtr sycl_target,
      std::vector<std::shared_ptr<CellwisePairList>> cellwise_pair_lists)
      : sycl_target(sycl_target), num_lists(cellwise_pair_lists.size()),
        h_pair_lists_device(
            std::vector<CellwisePairListDevice>(cellwise_pair_lists.size())),
        d_pair_lists_device(
            std::make_shared<BufferDevice<CellwisePairListDevice>>(
                sycl_target, cellwise_pair_lists.size())),
        cellwise_pair_lists(cellwise_pair_lists) {
    if (this->num_lists) {
      this->cell_count = cellwise_pair_lists.at(0)->cell_count;
      for (auto &cx : cellwise_pair_lists) {
        NESOASSERT(cell_count == cx->cell_count,
                   "Missmatched cell counts between lists.");
      }
    }
  }

  /**
   * Initialise wave creation using the current state of the pair lists.
   */
  inline void initialise_wave_creation() {
    if (this->num_lists) {

      int k_max_index = 0;
      for (std::size_t listx = 0; listx < this->num_lists; listx++) {
        this->h_pair_lists_device[listx] =
            this->cellwise_pair_lists[listx]->get();
        k_max_index =
            std::max(k_max_index, this->h_pair_lists_device[listx].max_index);
      }

      auto e0 = this->d_pair_lists_device->set_async(this->h_pair_lists_device);
      const auto k_pair_lists_device = this->d_pair_lists_device->ptr;

      const auto k_cell_count = this->cell_count;
      const auto k_num_lists = this->num_lists;

      auto d_wave_index = get_resource<BufferDevice<int>,
                                       ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);

      d_wave_index->realloc_no_copy(k_cell_count * k_num_lists * k_max_index);
      int *k_wave_index = d_wave_index->ptr;
      auto e1 = this->sycl_target->queue.fill<int>(
          k_wave_index, static_cast<int>(0),
          k_cell_count * k_num_lists * k_max_index);

      auto e2 = this->sycl_target->queue.parallel_for(
          sycl::range<2>(static_cast<std::size_t>(k_num_lists), k_cell_count),
          {e0, e1}, [=](sycl::item<2> idx) {
            const std::size_t index_list = idx.get_id(0);
            const std::size_t index_cell = idx.get_id(1);

            const CellwisePairListDevice *source_pair_list =
                &k_pair_lists_device[index_list];

            const int num_pairs_in_cell =
                source_pair_list->d_pair_counts[index_cell];

            for (int pairx = 0; pairx < num_pairs_in_cell; pairx++) {
              const int pi =
                  source_pair_list->d_pair_list[index_cell][0][pairx];
              const int pj =
                  source_pair_list->d_pair_list[index_cell][1][pairx];
            }
          });

      // TODO WAIT ON e2

      restore_resource(sycl_target->resource_stack_map,
                       ResourceStackKeyBufferDevice<int>{}, d_wave_index);
    }
  }

  /**
   * @returns Device copyable description of waves.
   */
  inline CellwisePairListWavesDevice get_waves() {
    CellwisePairListWavesDevice s{};

    if (this->num_lists) {
    }

    return s;
  }
};
} // namespace NESO::Particles::DSMC

#endif
