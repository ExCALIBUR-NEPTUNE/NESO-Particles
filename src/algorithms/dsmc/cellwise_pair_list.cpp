#include <neso_particles/algorithms/dsmc/cellwise_pair_list.hpp>

namespace NESO::Particles::DSMC {

CellwisePairList::CellwisePairList(SYCLTargetSharedPtr sycl_target,
                                   const int cell_count)
    : d_pair_list(std::make_shared<CellDat<int>>(sycl_target, cell_count, 2)),
      d_pair_counts(
          std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      sycl_target(sycl_target), cell_count(cell_count) {
  this->clear();
}

void CellwisePairList::push_back(const std::vector<int> &c,
                                 const std::vector<int> &i,
                                 const std::vector<int> &j) {

  const auto n = c.size();
  NESOASSERT(n == i.size(), "Input size missmatch.");
  NESOASSERT(n == j.size(), "Input size missmatch.");

  std::vector<int> layers(n);

  auto h_pair_counts = this->d_pair_counts->get();

  for (int ix = 0; ix < n; ix++) {
    NESOASSERT(0 <= c[ix] && c[ix] < this->cell_count, "Bad cell index.");
    layers[ix] = h_pair_counts[c[ix]];
    h_pair_counts[c[ix]]++;
  }

  for (int cx = 0; cx < cell_count; cx++) {
    const auto current_size = static_cast<int>(this->d_pair_list->nrow[cx]);
    const auto required_size = h_pair_counts[cx];
    if (required_size > current_size) {
      this->d_pair_list->set_nrow(cx, required_size);
    }
  }

  BufferDevice<int> d_c(this->sycl_target, c);
  BufferDevice<int> d_i(this->sycl_target, i);
  BufferDevice<int> d_j(this->sycl_target, j);
  BufferDevice<int> d_layers(this->sycl_target, layers);

  this->d_pair_list->wait_set_nrow();

  auto k_c = d_c.ptr;
  auto k_i = d_i.ptr;
  auto k_j = d_j.ptr;
  auto k_layers = d_layers.ptr;
  auto k_cell_list = this->d_pair_list->device_ptr();

  auto e0 =
      this->sycl_target->queue.parallel_for(sycl::range<1>(n), [=](auto idx) {
        const auto tc = k_c[idx];
        const auto ti = k_i[idx];
        const auto tj = k_j[idx];
        const auto tlayer = k_layers[idx];
        k_cell_list[tc][0][tlayer] = ti;
        k_cell_list[tc][1][tlayer] = tj;
      });

  this->d_pair_counts->set(h_pair_counts);
  e0.wait_and_throw();
}

void CellwisePairList::clear() {
  this->sycl_target->queue
      .fill(this->d_pair_counts->ptr, static_cast<int>(0), this->cell_count)
      .wait_and_throw();
}

CellWisePairListDevice CellwisePairList::get() {
  CellWisePairListDevice l = {this->d_pair_list->device_ptr(),
                              this->d_pair_counts->ptr};

  return l;
}

std::map<int, std::pair<std::vector<int>, std::vector<int>>>
CellwisePairList::host_get() {
  std::map<int, std::pair<std::vector<int>, std::vector<int>>> l;

  auto h_pair_counts = this->d_pair_counts->get();

  for (int cx = 0; cx < this->cell_count; cx++) {
    auto pairs = this->d_pair_list->get_cell(cx);
    const auto pair_count = h_pair_counts[cx];
    l[cx].first.resize(pair_count);
    l[cx].second.resize(pair_count);

    for (int rx = 0; rx < pair_count; rx++) {
      l[cx].first[rx] = pairs->at(rx, 0);
      l[cx].second[rx] = pairs->at(rx, 1);
    }
  }

  return l;
}
} // namespace NESO::Particles::DSMC
