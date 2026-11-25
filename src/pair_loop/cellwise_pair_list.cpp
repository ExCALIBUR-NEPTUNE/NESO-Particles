#include <neso_particles/pair_loop/cellwise_pair_list.hpp>

namespace NESO::Particles {

CellwisePairList::CellwisePairList(SYCLTargetSharedPtr sycl_target)
    : sycl_target(sycl_target) {}

bool CellwisePairList::validate_pair_list() { return false; }

INT CellwisePairList::get_num_pairs_range(const int cell_start,
                                          const int cell_end) {

  CellwisePairListDevice pair_list_device = this->get_pair_list();
  NESOASSERT(cell_start >= 0, "Bad cell_start: " + std::to_string(cell_start));
  NESOASSERT(cell_start < cell_end,
             "Bad cell_start<cell_end: " + std::to_string(cell_start) + " < " +
                 std::to_string(cell_end));
  NESOASSERT(cell_end <= pair_list_device.cell_count,
             "Bad cell_end: " + std::to_string(cell_end));

  // The h_pair_counts array in the pair list data structure is actually waves
  // then cells -> could reorder?

  INT count = 0;
  for (int cellx = cell_start; cellx < cell_end; cellx++) {
    const int num_waves = pair_list_device.get_num_waves_host(cellx);
    for (int wavex = 0; wavex < num_waves; wavex++) {
      count += pair_list_device.get_num_pairs_host(wavex, cellx);
    }
  }

  return count;
}

void CellwisePairList::get_num_pairs_masked_range_cellwise(
    const int cell_start, const int cell_end, int *d_num_pairs,
    EventStack &event_stack) {

  const auto k_pair_list = this->get_pair_list();

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  const int cell_range = cell_end - cell_start;
  const auto cell_count = k_pair_list.cell_count;

  NESOASSERT((0 <= cell_start) && (cell_start <= cell_count),
             "Bad cell_start.");
  NESOASSERT((0 <= cell_end) && (cell_end <= cell_count), "Bad cell_end.");

  NESOASSERT(d_num_pairs != nullptr, "Bad d_num_pairs.");
  auto e0 = this->sycl_target->queue.fill<int>(d_num_pairs, 0, cell_count);

  if (cell_range < 1) {
    event_stack.push(e0);
    return;
  }

  auto e1 = this->sycl_target->queue.parallel_for(
      this->sycl_target->device_limits.validate_nd_range(
          sycl::nd_range<2>(sycl::range<2>(cell_range, local_size),
                            sycl::range<2>(1, local_size))),
      e0, [=](sycl::nd_item<2> idx) {
        const int item_cell = cell_start + idx.get_global_id(0);
        const int item_local_id = idx.get_global_id(1);
        const auto &mask_array = k_pair_list.mask_array;

        int count_workitem = 0;
        const int num_waves = k_pair_list.get_num_waves(item_cell);
        for (int wavex = 0; wavex < num_waves; wavex++) {
          const int num_pairs = k_pair_list.get_num_pairs(wavex, item_cell);
          for (int pair_index = item_local_id; pair_index < num_pairs;
               pair_index += local_size) {
            const INT global_pair_index =
                k_pair_list.get_pair_linear_index(wavex, item_cell, pair_index);

            const bool mask = mask_array.get(global_pair_index, 0);
            count_workitem += mask ? 1 : 0;
          }
        }

        // TODO use group reduce (acpp bugs)?
        atomic_fetch_add(d_num_pairs + item_cell, count_workitem);
      });

  event_stack.push(e1);
}

} // namespace NESO::Particles
