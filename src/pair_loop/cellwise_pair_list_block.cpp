#include <neso_particles/error_propagate.hpp>
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

bool CellwisePairListBlockInterface::validate_pair_list(
    SYCLTargetSharedPtr sycl_target) {
  auto d_pair_list = this->get_pair_list();

  const int pair_count = d_pair_list.pair_count;
  const int cell_count = d_pair_list.cell_count;
  const int max_wave_count = d_pair_list.max_wave_count;

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  const auto block_size = d_pair_list.block_size;

  BufferDevice<int> d_pair_count(sycl_target, 1);
  auto h_pair_count = d_pair_count.get();
  h_pair_count[0] = 0;
  d_pair_count.set(h_pair_count);

  int *k_pair_count = d_pair_count.ptr;

  if (max_wave_count > block_size) {
    nprint("bad max wave count:", max_wave_count, block_size);
    return false;
  }

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 1> la_i(sycl::range<1>(block_size), cgh);
        sycl::local_accessor<int, 1> la_j(sycl::range<1>(block_size), cgh);
        sycl::local_accessor<int, 1> la_w(sycl::range<1>(block_size), cgh);

        cgh.parallel_for(
            sycl_target->device_limits.validate_nd_range(sycl::nd_range<1>(
                sycl::range<1>(cell_count), sycl::range<1>(1))),
            [=](sycl::nd_item<1> idx) {
              const int index_cell = idx.get_global_id(0);

              const int num_pairs = d_pair_list.get_num_pairs(index_cell);

              for (int ix = 0; ix < block_size; ix++) {
                la_i[ix] = -1;
                la_j[ix] = -1;
              }

              if (index_cell == 0) {
                NESO_KERNEL_ASSERT(d_pair_list.get_pair_linear_index(0, 0) == 0,
                                   k_ep);
              }

              if (index_cell < (cell_count - 1)) {
                const INT c0 = d_pair_list.get_pair_linear_index(index_cell, 0);
                const INT c1 =
                    d_pair_list.get_pair_linear_index(index_cell + 1, 0);
                const int diff = static_cast<int>(c1 - c0);

                NESO_KERNEL_ASSERT(
                    d_pair_list.d_pair_counts[index_cell] == diff, k_ep);
              }

              const int num_blocks = div_round_up(num_pairs, block_size);
              int pair_index_offset = 0;
              for (int block = 0; block < num_blocks; block++) {
                const int num_waves =
                    d_pair_list.get_num_waves(index_cell, block);

                int max_seen_wave = 0;

                for (int index_local = 0; index_local < block_size;
                     index_local++) {
                  const int pair_index = pair_index_offset + index_local;
                  const bool required = pair_index < num_pairs;
                  const int i =
                      required
                          ? d_pair_list.get_pair_index_i(index_cell, pair_index)
                          : -2;
                  const int j =
                      required
                          ? d_pair_list.get_pair_index_j(index_cell, pair_index)
                          : -3;
                  NESO_KERNEL_ASSERT(i != j, k_ep);
                  const int wave =
                      required
                          ? d_pair_list.get_pair_wave(index_cell, pair_index)
                          : -4;

                  if (required) {
                    NESO_KERNEL_ASSERT((0 <= wave) && (wave < max_wave_count),
                                       k_ep);
                    NESO_KERNEL_ASSERT((0 <= wave) && (wave < num_waves), k_ep);
                    max_seen_wave = sycl::max(wave, max_seen_wave);
                  }

                  la_i[index_local] = i;
                  la_j[index_local] = j;
                  la_w[index_local] = wave;
                }

                NESO_KERNEL_ASSERT(max_seen_wave + 1 == num_waves, k_ep);

                for (int ix = 0; ix < block_size; ix++) {
                  const int i0 = la_i[ix];
                  const int j0 = la_j[ix];
                  const int w0 = la_w[ix];
                  if (w0 > -1) {
                    atomic_fetch_add(k_pair_count, 1);
                    for (int jx = ix + 1; jx < block_size; jx++) {
                      const int i1 = la_i[jx];
                      const int j1 = la_j[jx];
                      const int w1 = la_w[jx];
                      if (w0 == w1) {
                        NESO_KERNEL_ASSERT(i0 != i1, k_ep);
                        NESO_KERNEL_ASSERT(j0 != j1, k_ep);
                        NESO_KERNEL_ASSERT(i0 != j1, k_ep);
                        NESO_KERNEL_ASSERT(j0 != i1, k_ep);
                      }
                    }
                  }
                }
                pair_index_offset += block_size;
              }
            });
      })
      .wait_and_throw();

  if (d_pair_count.get().at(0) != pair_count) {
    nprint("pair count failed", pair_count, d_pair_count.get().at(0));
    return false;
  }

  if (ep.get_flag()) {
    nprint("get_flag failed");
    return false;
  }

  return true;
}

void CellwisePairListBlockInterface::get_wave_occupancy_counts(
    SYCLTargetSharedPtr sycl_target, std::vector<int> &occupancy_counts) {

  auto d_pair_list = this->get_pair_list();
  const int pair_count = d_pair_list.pair_count;
  const int cell_count = d_pair_list.cell_count;
  const int max_wave_count = d_pair_list.max_wave_count;
  const auto block_size = d_pair_list.block_size;

  occupancy_counts.resize(block_size + 1);
  std::fill(occupancy_counts.begin(), occupancy_counts.end(), 0);

  auto d_occupancy_counts =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_occupancy_counts->realloc_no_copy(block_size + 1);
  int *k_occupancy_counts = d_occupancy_counts->ptr;

  sycl_target->queue
      .memcpy(k_occupancy_counts, occupancy_counts.data(),
              (block_size + 1) * sizeof(int))
      .wait_and_throw();

  auto e1 = sycl_target->queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> la_occupancy_counts(
        sycl::range<1>(block_size + 1), cgh);
    sycl::local_accessor<int, 1> la_waves(sycl::range<1>(block_size), cgh);

    cgh.parallel_for(
        sycl_target->device_limits.validate_nd_range(
            sycl::nd_range<2>(sycl::range<2>(cell_count, block_size),
                              sycl::range<2>(1, block_size))),
        [=](sycl::nd_item<2> idx) {
          const int index_cell = idx.get_global_id(0);
          const int index_local = idx.get_global_id(1);
          la_occupancy_counts[index_local] = 0;
          if (index_local == (block_size - 1)) {
            la_occupancy_counts[block_size] = 0;
          }
          idx.barrier(sycl::access::fence_space::local_space);

          const int num_pairs = d_pair_list.get_num_pairs(index_cell);
          const int num_blocks = div_round_up(num_pairs, block_size);

          int index_pair = index_local;
          for (int index_block = 0; index_block < num_blocks; index_block++) {
            int wave = -1;
            if (index_pair < num_pairs) {
              wave = d_pair_list.get_pair_wave(index_cell, index_pair);
            }
            la_waves[index_local] = wave;

            const int num_waves =
                d_pair_list.get_num_waves(index_cell, index_block);

            idx.barrier(sycl::access::fence_space::local_space);
            int wave_count = 0;
            for (int ix = 0; ix < block_size; ix++) {
              if (la_waves[ix] == index_local) {
                wave_count++;
              }
            }

            // If the wave index is less than the number of waves in the block
            // then an empty wave does contribute to the dispatch count and
            // affect the occupancy.
            if ((index_local < num_waves)) {
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group>(
                  la_occupancy_counts[wave_count])
                  .fetch_add(1);
            }

            idx.barrier(sycl::access::fence_space::local_space);
            index_pair += block_size;
          }

          atomic_fetch_add(k_occupancy_counts + index_local,
                           la_occupancy_counts[index_local]);
          if (index_local == (block_size - 1)) {
            atomic_fetch_add(k_occupancy_counts + block_size,
                             la_occupancy_counts[block_size]);
          }
        });
  });

  // N.B. this call passes e1 hence we don't explicitly wait on e1 here.
  sycl_target->queue
      .memcpy(occupancy_counts.data(), k_occupancy_counts,
              (block_size + 1) * sizeof(int), e1)
      .wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_occupancy_counts);
}

REAL get_mean_wave_occupancy(std::vector<int> &occupancy_counts) {

  const int block_size = static_cast<int>(occupancy_counts.size()) - 1;
  REAL occupancy_unscaled = 0.0;

  int num_blocks = 0.0;
  for (int wave_size = 0; wave_size < (block_size + 1); wave_size++) {
    occupancy_unscaled += wave_size * occupancy_counts[wave_size];
    num_blocks += occupancy_counts[wave_size];
  }

  if (num_blocks == 0) {
    return 0.0;
  } else {
    const REAL occupancy =
        occupancy_unscaled / static_cast<REAL>(block_size * num_blocks);
    return occupancy;
  }
}

void get_global_wave_occupancy_counts(
    SYCLTargetSharedPtr sycl_target, std::vector<int> &local_occupancy_counts,
    std::vector<int> &global_occupancy_counts) {

  MPI_Comm comm = sycl_target->comm_pair.comm_parent;

  const int num_values_in = static_cast<int>(local_occupancy_counts.size());
  int num_values = num_values_in;

  MPICHK(MPI_Bcast(&num_values, 1, MPI_INT, 0, comm));
  NESOASSERT(num_values == num_values_in,
             "Missmatch in input size across ranks.");

  const int rank = sycl_target->comm_pair.rank_parent;
  if (rank && (!global_occupancy_counts.size())) {
    global_occupancy_counts.resize(1);
  } else {
    global_occupancy_counts.resize(num_values);
  }

  MPICHK(MPI_Reduce(local_occupancy_counts.data(),
                    global_occupancy_counts.data(), num_values, MPI_INT,
                    MPI_SUM, 0, comm));

  // We do this resize after the MPI call as some MPI implementations
  // complain if they are passed nullptrs (even if they have no reason to
  // dereference them).
  if (rank) {
    global_occupancy_counts.resize(0);
  }
}

} // namespace NESO::Particles
