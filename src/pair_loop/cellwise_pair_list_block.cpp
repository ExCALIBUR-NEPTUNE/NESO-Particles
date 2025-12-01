#include <neso_particles/pair_loop/cellwise_pair_list_block.hpp>

namespace NESO::Particles {

std::map<int, std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>>
CellwisePairListBlockInterface::get_host_pair_list(
    SYCLTargetSharedPtr sycl_target) {

  std::map<int,
           std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>>
      h_pair_list;
  auto d_pair_list = this->get_pair_list();
  const int pair_count = d_pair_list.pair_count;

  const int cell_count = d_pair_list.cell_count;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    h_pair_list[cellx] = {{}, {}, {}};
  }

  std::vector<int> h_pair_counts(cell_count);
  sycl_target->queue
      .memcpy(h_pair_counts.data(), d_pair_list.d_pair_counts,
              cell_count * sizeof(int))
      .wait_and_throw();

  std::vector<int> pairs_i(pair_count);
  std::vector<int> pairs_j(pair_count);
  std::vector<int> waves(pair_count);

  sycl_target->queue
      .memcpy(pairs_i.data(), d_pair_list.d_pair_list, pair_count * sizeof(int))
      .wait_and_throw();

  sycl_target->queue
      .memcpy(pairs_j.data(), d_pair_list.d_pair_list + pair_count,
              pair_count * sizeof(int))
      .wait_and_throw();

  sycl_target->queue
      .memcpy(waves.data(), d_pair_list.d_pair_list + pair_count * 2,
              pair_count * sizeof(int))
      .wait_and_throw();

  int offset = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int cell_pair_count = h_pair_counts[cellx];
    std::get<0>(h_pair_list[cellx]).resize(cell_pair_count);
    std::get<1>(h_pair_list[cellx]).resize(cell_pair_count);
    std::get<2>(h_pair_list[cellx]).resize(cell_pair_count);

    for (int ix = 0; ix < cell_pair_count; ix++) {
      std::get<0>(h_pair_list[cellx]).at(ix) = pairs_i.at(offset + ix);
      std::get<1>(h_pair_list[cellx]).at(ix) = pairs_j.at(offset + ix);
      std::get<2>(h_pair_list[cellx]).at(ix) = waves.at(offset + ix);
    }

    offset += cell_pair_count;
  }

  return h_pair_list;
}

} // namespace NESO::Particles
