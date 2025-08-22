#include "include/test_neso_particles.hpp"

TEST(DSMC, cellwise_pair_list) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int cell_count = 19;
  auto cellwise_pair_list =
      std::make_shared<DSMC::CellwisePairList>(sycl_target, cell_count);

  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  std::mt19937 rng(522342);
  std::uniform_int_distribution<int> dist(0, cell_count - 1);

  const int num_samples = 100;

  std::map<int, std::pair<std::vector<int>, std::vector<int>>> h_correct;
  std::vector<int> h_c(num_samples);
  std::vector<int> h_i(num_samples);
  std::vector<int> h_j(num_samples);

  for (int ix = 0; ix < num_samples; ix++) {
    h_c[ix] = dist(rng);
    h_i[ix] = dist(rng);
    h_j[ix] = dist(rng);
    h_correct[h_c[ix]].first.push_back(h_i[ix]);
    h_correct[h_c[ix]].second.push_back(h_j[ix]);
  }

  cellwise_pair_list->push_back(h_c, h_i, h_j);

  {
    auto h_to_test = cellwise_pair_list->host_get();

    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_correct[cx], h_to_test[cx]);
    }
  }

  for (int ix = 0; ix < num_samples; ix++) {
    h_c[ix] = dist(rng);
    h_i[ix] = dist(rng);
    h_j[ix] = dist(rng);
    h_correct[h_c[ix]].first.push_back(h_i[ix]);
    h_correct[h_c[ix]].second.push_back(h_j[ix]);
  }

  cellwise_pair_list->push_back(h_c, h_i, h_j);

  {
    auto h_to_test = cellwise_pair_list->host_get();

    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_correct[cx], h_to_test[cx]);
    }
  }

  cellwise_pair_list->clear();
  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  {
    h_c.resize(0);
    h_i.resize(0);
    h_j.resize(0);
    cellwise_pair_list->push_back(h_c, h_i, h_j);
  }

  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  sycl_target->free();
}

struct CellwiseAllToAll {
  const int workgroup_size{0};
  const int workgroup_index{0};

  CellwiseAllToAll(const int workgroup_size, const int workgroup_index)
      : workgroup_size(workgroup_size), workgroup_index(workgroup_index) {}

  template <typename KERNEL_FUNC_TYPE, typename SYNC_FUNC_TYPE>
  inline void
  apply_inner_block(const int workgroup_block_item,
                    const int workgroup_block_size, const int block_start_row,
                    const int block_start_col, const int n,
                    KERNEL_FUNC_TYPE &kernel_func, SYNC_FUNC_TYPE &sync_func,
                    const bool mask_required) const {

    int index = workgroup_block_item;
    const int mask = workgroup_block_size - 1;
    if (mask_required) {
      for (int rowx = block_start_row;
           rowx < (block_start_row + workgroup_block_size); rowx++) {
        const int colx = index + block_start_col;
        if ((rowx < n) && (colx < n)) {
          kernel_func(rowx, colx);
        }
        sync_func();
        index = (index + 1) & mask;
      }
    } else {
      for (int rowx = block_start_row;
           rowx < (block_start_row + workgroup_block_size); rowx++) {
        const int colx = index + block_start_col;
        kernel_func(rowx, colx);
        sync_func();
        index = (index + 1) & mask;
      }
    }
  }
  template <typename KERNEL_FUNC_TYPE, typename SYNC_FUNC_TYPE>
  inline void apply_diagonal(const int n, const int block_offset,
                             KERNEL_FUNC_TYPE &kernel_func,
                             SYNC_FUNC_TYPE &sync_func,
                             const bool mask_required) const {
    int size = this->workgroup_size;
    while (size > 0) {
      const int workgroup_block = this->workgroup_index / size;
      const int workgroup_block_item = this->workgroup_index % size;

      int block_end_row = size + workgroup_block * 2 * size;
      int block_start_row = block_end_row - size;
      int block_start_col = block_end_row;

      block_start_row += block_offset;
      block_start_col += block_offset;

      this->apply_inner_block(workgroup_block_item, size, block_start_row,
                              block_start_col, n, kernel_func, sync_func,
                              mask_required);

      size /= 2;
    }
  }
  template <typename KERNEL_FUNC_TYPE, typename SYNC_FUNC_TYPE>
  inline void apply(const int n, KERNEL_FUNC_TYPE kernel_func,
                    SYNC_FUNC_TYPE sync_func) const {
    const int block_size_inner = this->workgroup_size;
    const int block_size = this->workgroup_size * 2;
    const auto n_padded = get_next_multiple(n, block_size);
    const int num_blocks = n_padded / block_size;

    for (int blockx = 0; blockx < num_blocks; blockx++) {

      const int block_offset = blockx * block_size;
      const bool masking_diagonal = blockx == (num_blocks - 1);
      // Process the diagonal blocks. The last block needs masking.
      apply_diagonal(n, block_offset, kernel_func, sync_func, masking_diagonal);

      // process the rest of the blocks in the remainder of the row
      const int row_start = block_offset;
      const int row_end = block_offset + block_size;
      const int col_start = block_offset + block_size;
      const int col_end = n_padded;

      for (int rowx = row_start; rowx < row_end; rowx += block_size_inner) {
        for (int colx = col_start; colx < col_end; colx += block_size_inner) {
          apply_inner_block(this->workgroup_index, this->workgroup_size, rowx,
                            colx, n, kernel_func, sync_func,
                            ((colx + block_size_inner) > n) ||
                                masking_diagonal);
        }
      }
    }
  }
};

TEST(DSMC, all_to_all_looping_host) {

  const int workgroup_size = 4;

  for (int num_particles : {0, 1, 3, 4, 7, 8, 31}) {

    std::vector<int> seen_entries(num_particles * num_particles);
    std::fill(seen_entries.begin(), seen_entries.end(), 0);
    std::vector<int> num_sync_calls(workgroup_size);
    std::fill(num_sync_calls.begin(), num_sync_calls.end(), 0);

    for (int workgroup_item = 0; workgroup_item < workgroup_size;
         workgroup_item++) {
      auto loop_context = CellwiseAllToAll(workgroup_size, workgroup_item);

      loop_context.apply(
          num_particles,
          [&](auto i, auto j) {
            seen_entries.at(i + num_particles * j)++;
            seen_entries.at(j + num_particles * i)++;
          },
          [&]() { num_sync_calls.at(workgroup_item)++; });
    }

    for (int ix = 0; ix < workgroup_size; ix++) {
      ASSERT_EQ(num_sync_calls.at(ix), num_sync_calls.at(0));
    }

    for (int rowx = 0; rowx < num_particles; rowx++) {
      for (int colx = 0; colx < num_particles; colx++) {
        ASSERT_EQ(seen_entries.at(rowx + num_particles * colx),
                  rowx == colx ? 0 : 1);
      }
    }
  }
}

TEST(DSMC, all_to_all_looping_device) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

#ifdef __ACPP__
  const int cpu_workgroup_size = 1;
  const bool use_barrier = sycl_target->device.is_cpu() ? false : true;
#else
  const int cpu_workgroup_size = 32;
  constexpr bool use_barrier = true;
#endif

  const int workgroup_size =
      sycl_target->device.is_cpu() ? cpu_workgroup_size : 256;

  for (int num_particles : {0, 1, 3, 4, 7, 8, 31, 255, 1023, 1024}) {

    std::vector<int> seen_entries(num_particles * num_particles);
    std::fill(seen_entries.begin(), seen_entries.end(), 0);
    BufferDevice<int> d_seen_entries(sycl_target, seen_entries);
    auto k_seen_entries = d_seen_entries.ptr;

    sycl_target->queue
        .parallel_for(sycl::nd_range<1>(sycl::range<1>(workgroup_size),
                                        sycl::range<1>(workgroup_size)),
                      [=](sycl::nd_item<1> idx) {
                        CellwiseAllToAll loop_context(idx.get_local_range(0),
                                                      idx.get_local_id(0));

                        loop_context.apply(
                            num_particles,
                            [&](auto i, auto j) {
                              k_seen_entries[i + num_particles * j]++;
                              k_seen_entries[j + num_particles * i]++;
                            },
                            [=]() {
                              if (use_barrier) {
                                sycl::group_barrier(idx.get_group());
                              }
                            });
                      })
        .wait_and_throw();

    seen_entries = d_seen_entries.get();

    for (int rowx = 0; rowx < num_particles; rowx++) {
      for (int colx = 0; colx < num_particles; colx++) {
        ASSERT_EQ(seen_entries.at(rowx + num_particles * colx),
                  rowx == colx ? 0 : 1);
      }
    }
  }

  sycl_target->free();
}
