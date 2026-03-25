#include <neso_particles/algorithms/common.hpp>
#include <neso_particles/pair_loop/cellwise_pair_list_simple.hpp>

namespace NESO::Particles {

CellwisePairListSimple::CellwisePairListSimple(SYCLTargetSharedPtr sycl_target,
                                               const int cell_count)
    : h_wave_count(std::vector<int>(cell_count)),
      d_wave_count(
          std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      d_wave_offsets(
          std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      d_pair_list(std::make_shared<CellDat<int>>(sycl_target, cell_count, 2)),
      d_pair_counts(
          std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      d_pair_counts_es(
          std::make_shared<BufferDevice<INT>>(sycl_target, cell_count)),
      h_pair_counts(std::vector<int>(cell_count)), sycl_target(sycl_target),
      cell_count(cell_count) {
  this->clear();
}

void CellwisePairListSimple::push_back(const std::vector<int> &c,
                                       const std::vector<int> &i,
                                       const std::vector<int> &j) {

  NESOASSERT((this->mode == 0) || (this->mode == 1),
             "push_back and push_back_waves cannot be used at the same time");
  this->mode = 1;

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

  auto e4 = this->sycl_target->queue.fill(
      this->d_wave_count->ptr, static_cast<int>(1), this->cell_count);
  std::fill(this->h_wave_count.begin(), this->h_wave_count.end(), 1);
  this->max_wave_count = 1;

  e0.wait_and_throw();
  e4.wait_and_throw();
}

void CellwisePairListSimple::set(CellwisePairListHostSharedPtr pair_list) {
  NESOASSERT(this->cell_count == pair_list->cell_count,
             "Cell count missmatch.");

  auto &m = pair_list->get();

  EventStack event_stack;
  this->max_wave_count = 0;
  {
    for (int cellx = 0; cellx < this->cell_count; cellx++) {
      const int tmp_wave_count = static_cast<int>(m[cellx].size());
      this->h_wave_count[cellx] = tmp_wave_count;
      this->max_wave_count = std::max(this->max_wave_count, tmp_wave_count);
    }
  }

  {
    event_stack.push(this->d_wave_count->set_async(this->h_wave_count));
    this->h_wave_offsets.resize(max_wave_count * this->cell_count);
    this->d_wave_offsets->realloc_no_copy(max_wave_count * this->cell_count);
    this->h_pair_counts.resize(max_wave_count * this->cell_count);
    this->d_pair_counts->realloc_no_copy(max_wave_count * this->cell_count);
    this->h_pair_counts_es.resize(max_wave_count * this->cell_count);
    this->d_pair_counts_es->realloc_no_copy(max_wave_count * this->cell_count);
  }

  auto k_wave_offsets = this->d_wave_offsets->ptr;
  auto k_pair_counts = this->d_pair_counts->ptr;
  auto k_pair_counts_es = this->d_pair_counts_es->ptr;

  {
    std::fill(this->h_pair_counts.begin(), this->h_pair_counts.end(), 0);
    std::fill(this->h_wave_offsets.begin(), this->h_wave_offsets.end(), 0);
  }

  {

    INT linear_offset = 0;
    for (int cellx = 0; cellx < this->cell_count; cellx++) {
      const int wave_count = m[cellx].size();

      int num_pairs = 0;
      for (int wavex = 0; wavex < wave_count; wavex++) {
        const int num_pairs_wave = m[cellx].at(wavex).first.size();
        this->h_pair_counts.at(wavex * cell_count + cellx) = num_pairs_wave;
        this->h_wave_offsets.at(wavex * cell_count + cellx) = num_pairs;
        // We might want to swap the ordering of linear_offset here to make
        // cells faster than waves.
        this->h_pair_counts_es.at(wavex * cell_count + cellx) = linear_offset;
        num_pairs += num_pairs_wave;
        linear_offset += num_pairs_wave;
      }
      if (num_pairs) {
        this->d_pair_list->set_nrow(cellx, num_pairs);
      }
    }

    if (max_wave_count) {
      event_stack.push(this->sycl_target->queue.memcpy(
          k_wave_offsets, this->h_wave_offsets.data(),
          max_wave_count * this->cell_count * sizeof(int)));
      event_stack.push(this->sycl_target->queue.memcpy(
          k_pair_counts, this->h_pair_counts.data(),
          max_wave_count * this->cell_count * sizeof(int)));
      event_stack.push(this->sycl_target->queue.memcpy(
          k_pair_counts_es, this->h_pair_counts_es.data(),
          max_wave_count * this->cell_count * sizeof(INT)));
    }
  }

  {
    this->d_pair_list->wait_set_nrow();

    this->max_pair_count = 0;
    this->pair_count = 0;
    for (int cellx = 0; cellx < this->cell_count; cellx++) {
      const int wave_count = m[cellx].size();

      int *particle_i_ptr = this->d_pair_list->col_device_ptr(cellx, 0);
      int *particle_j_ptr = this->d_pair_list->col_device_ptr(cellx, 1);

      int num_pairs = 0;
      for (int wavex = 0; wavex < wave_count; wavex++) {
        const int num_pairs_wave = m[cellx].at(wavex).first.size();
        if (num_pairs_wave) {
          const int *source_i_ptr = m[cellx].at(wavex).first.data();
          const int *source_j_ptr = m[cellx].at(wavex).second.data();
          this->max_pair_count = std::max(this->max_pair_count, num_pairs_wave);
          this->pair_count += num_pairs_wave;

          event_stack.push(this->sycl_target->queue.memcpy(
              particle_i_ptr + num_pairs, source_i_ptr,
              num_pairs_wave * sizeof(int)));

          event_stack.push(this->sycl_target->queue.memcpy(
              particle_j_ptr + num_pairs, source_j_ptr,
              num_pairs_wave * sizeof(int)));
        }
        num_pairs += num_pairs_wave;
      }
    }
  }

  event_stack.wait();
}

void CellwisePairListSimple::clear() {
  if (this->cell_count > 0) {
    auto k_wave_count = this->d_wave_count->ptr;
    auto k_pair_counts = this->d_pair_counts->ptr;
    auto k_wave_offsets = this->d_wave_offsets->ptr;

    auto e0 = this->sycl_target->queue.parallel_for(
        sycl::range<1>(this->cell_count), [=](auto ix) {
          k_wave_count[ix] = 0;
          k_pair_counts[ix] = 0;
          k_wave_offsets[ix] = 0;
        });

    std::fill(this->h_wave_count.begin(), this->h_wave_count.end(), 0);
    std::fill(this->h_pair_counts.begin(), this->h_pair_counts.end(), 0);
    e0.wait_and_throw();
  }
  this->max_pair_count = -1;
  this->max_wave_count = -1;
  this->pair_count = 0;
  this->mode = 0;
}

CellwisePairListDevice CellwisePairListSimple::get_pair_list() {
  CellwisePairListDevice l = {this->h_wave_count.data(),
                              this->d_wave_count->ptr,
                              this->cell_count,
                              this->d_wave_offsets->ptr,
                              this->d_pair_list->device_ptr(),
                              this->d_pair_counts->ptr,
                              this->d_pair_counts_es->ptr,
                              this->h_pair_counts.data(),
                              this->max_pair_count,
                              this->max_wave_count,
                              this->pair_count};

  return l;
}

CellwisePairListHostMap CellwisePairListSimple::get_host_pair_list() {
  CellwisePairListHostMap l;

  for (int cx = 0; cx < this->cell_count; cx++) {
    auto pairs = this->d_pair_list->get_cell(cx);

    const int wave_count = this->h_wave_count.at(cx);
    int index = 0;
    for (int wavex = 0; wavex < wave_count; wavex++) {
      const auto pair_count = this->h_pair_counts[wavex * cell_count + cx];
      for (int px = 0; px < pair_count; px++) {
        l[cx][wavex].first.push_back(pairs->at(index, 0));
        l[cx][wavex].second.push_back(pairs->at(index, 1));
        index++;
      }
    }
  }

  return l;
}

INT CellwisePairListSimple::get_num_pairs() { return this->pair_count; }

} // namespace NESO::Particles
