#include <neso_particles/pair_loop/cellwise_pair_list.hpp>

namespace NESO::Particles {

CellwisePairList::CellwisePairList(SYCLTargetSharedPtr sycl_target,
                                   const int cell_count)
    : d_pair_list(std::make_shared<CellDat<int>>(sycl_target, cell_count, 2)),
      d_pair_counts(
          std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      d_pair_counts_es(
          std::make_shared<BufferDevice<INT>>(sycl_target, cell_count)),
      h_pair_counts(std::vector<int>(cell_count)), sycl_target(sycl_target),
      cell_count(cell_count) {
  this->clear();
}

void CellwisePairList::push_back(const std::vector<int> &c,
                                 const std::vector<int> &i,
                                 const std::vector<int> &j) {

  const auto n = c.size();
  NESOASSERT(n == i.size(), "Input size missmatch.");
  NESOASSERT(n == j.size(), "Input size missmatch.");
  if (n == 0) {
    return;
  }

  std::vector<int> layers(n);
  this->d_pair_counts->get(this->h_pair_counts);

  this->max_pair_count = -1;
  for (int ix = 0; ix < n; ix++) {
    NESOASSERT(0 <= c[ix] && c[ix] < this->cell_count, "Bad cell index.");
    layers[ix] = this->h_pair_counts[c[ix]];
    this->h_pair_counts[c[ix]]++;
    this->max_pair_count =
        std::max(this->max_pair_count, this->h_pair_counts[c[ix]]);
  }

  for (int cx = 0; cx < cell_count; cx++) {
    const auto current_size = static_cast<int>(this->d_pair_list->nrow[cx]);
    const auto required_size = this->h_pair_counts[cx];
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

  BufferDevice<int> d_max_indices(this->sycl_target, 2);

  auto e1 = reduce_values(this->sycl_target, i.size(), d_i.ptr,
                          sycl::maximum<int>(), d_max_indices.ptr);

  auto e2 = reduce_values(this->sycl_target, j.size(), d_j.ptr,
                          sycl::maximum<int>(), d_max_indices.ptr + 1);

  this->d_pair_counts->set(this->h_pair_counts);
  auto k_pair_counts = this->d_pair_counts->ptr;

  auto d_counts =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_counts->realloc_no_copy(this->cell_count);
  INT *k_counts = d_counts->ptr;

  this->sycl_target->queue
      .parallel_for(sycl::range<1>(this->cell_count),
                    [=](auto idx) {
                      k_counts[idx] = static_cast<INT>(k_pair_counts[idx]);
                    })
      .wait_and_throw();

  auto e3 = joint_exclusive_scan(this->sycl_target, this->cell_count, k_counts,
                                 this->d_pair_counts_es->ptr);
  e3.wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_counts);

  // Get the total number of pairs
  INT last_es_count = -1;
  this->sycl_target->queue
      .memcpy(&last_es_count,
              this->d_pair_counts_es->ptr + this->cell_count - 1, sizeof(INT))
      .wait_and_throw();

  NESOASSERT(last_es_count >= 0, "Bad last cell count.");

  this->pair_count = last_es_count + this->h_pair_counts[this->cell_count - 1];

  e0.wait_and_throw();
  e1.wait_and_throw();
  e2.wait_and_throw();

  auto h_max_indicies = d_max_indices.get();
  this->max_index = std::max(h_max_indicies[0], h_max_indicies[1]);
}

void CellwisePairList::clear() {
  if (this->cell_count > 0) {
    auto e0 = this->sycl_target->queue.fill(
        this->d_pair_counts->ptr, static_cast<int>(0), this->cell_count);
    std::fill(this->h_pair_counts.begin(), this->h_pair_counts.end(), 0);
    e0.wait_and_throw();
  }
  this->max_index = -1;
  this->max_pair_count = -1;
  this->pair_count = -1;
}

CellwisePairListDevice CellwisePairList::get() {
  CellwisePairListDevice l = {
      this->cell_count,           this->d_pair_list->device_ptr(),
      this->d_pair_counts->ptr,   this->d_pair_counts_es->ptr,
      this->h_pair_counts.data(), this->max_index,
      this->max_pair_count,       this->pair_count};

  return l;
}

std::map<int, std::pair<std::vector<int>, std::vector<int>>>
CellwisePairList::host_get() {
  std::map<int, std::pair<std::vector<int>, std::vector<int>>> l;

  for (int cx = 0; cx < this->cell_count; cx++) {
    auto pairs = this->d_pair_list->get_cell(cx);
    const auto pair_count = this->h_pair_counts[cx];
    l[cx].first.resize(pair_count);
    l[cx].second.resize(pair_count);

    for (int rx = 0; rx < pair_count; rx++) {
      l[cx].first[rx] = pairs->at(rx, 0);
      l[cx].second[rx] = pairs->at(rx, 1);
    }
  }

  return l;
}
} // namespace NESO::Particles
