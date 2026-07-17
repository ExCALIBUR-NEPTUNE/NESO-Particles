#include <neso_particles/pair_loop/cellwise_pair_list.hpp>

namespace NESO::Particles {

bool CellwisePairList::validate_pair_list(
    [[maybe_unused]] SYCLTargetSharedPtr sycl_target) {

  return false;
}

INT CellwisePairList::get_num_pairs_range(const int cell_start,
                                          const int cell_end) {

  CellwisePairListDevice pair_list_device = this->get_pair_list();
  NESOASSERT(cell_start >= 0, "Bad cell_start: " + std::to_string(cell_start));
  NESOASSERT(cell_start < cell_end,
             "Bad cell_start<cell_end: " + std::to_string(cell_start) + " < " +
                 std::to_string(cell_end));
  NESOASSERT(cell_end < pair_list_device.cell_count,
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

} // namespace NESO::Particles
