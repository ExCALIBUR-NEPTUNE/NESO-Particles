#include <neso_particles/algorithms/dsmc/pair_sampler_no_replacement.hpp>

namespace NESO::Particles::DSMC {

PairSamplerNoReplacement::PairSamplerNoReplacement(
    SYCLTargetSharedPtr sycl_target, const int cell_count,
    std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function)
    : sycl_target(sycl_target), cell_count(cell_count),
      rng_generation_function(rng_generation_function) {

  this->d_pair_list =
      std::make_unique<CellDat<int>>(this->sycl_target, this->cell_count, 2);

  this->h_wave_count.resize(this->cell_count);
  std::fill(this->h_wave_count.begin(), this->h_wave_count.end(), 1);
  this->d_wave_count =
      std::make_unique<BufferDevice<int>>(sycl_target, this->h_wave_count);

  this->h_pair_counts.resize(this->cell_count);
  this->d_pair_counts =
      std::make_unique<BufferDevice<int>>(this->sycl_target, this->cell_count);

  this->h_pair_counts_es.resize(this->cell_count);
  this->d_pair_counts_es =
      std::make_unique<BufferDevice<INT>>(this->sycl_target, this->cell_count);

  this->h_num_collision_cells.resize(this->cell_count);
  this->d_num_collision_cells =
      std::make_unique<BufferDevice<int>>(this->sycl_target, this->cell_count);
}

void PairSamplerNoReplacement::sample(
    CollisionCellPartitionSharedPtr collision_cell_partition,
    const INT species_id_a, const INT species_id_b,
    const std::vector<std::vector<int>> &map_cells_to_counts) {
  NESOASSERT(this->cell_count == collision_cell_partition->cell_count,
             "Cell count missmatch.");

  const INT max_num_collision_cells =
      collision_cell_partition->max_num_collision_cells;

  auto d_pair_counts_ccell =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_pair_counts_ccell->realloc_no_copy(max_num_collision_cells *
                                       this->cell_count);
  auto k_pair_counts_ccell = d_pair_counts_ccell->ptr;

  auto d_pair_counts_ccell_es =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_pair_counts_ccell_es->realloc_no_copy(max_num_collision_cells *
                                          this->cell_count);
  auto k_pair_counts_ccell_es = d_pair_counts_ccell_es->ptr;

  EventStack es;
  for (INT mx = 0; mx < this->cell_count; mx++) {
    const auto num_collision_cells = map_cells_to_counts.at(mx).size();
    NESOASSERT(num_collision_cells <= max_num_collision_cells,
               "More cell counts than collision cells passed.");

    es.push(this->sycl_target->queue.memcpy(
        k_pair_counts_ccell + mx * max_num_collision_cells,
        map_cells_to_counts[mx].data(), num_collision_cells * sizeof(int)));

    this->h_num_collision_cells.at(mx) = static_cast<int>(num_collision_cells);
  }

  auto k_num_collision_cells = this->d_num_collision_cells->ptr;
  es.push(this->sycl_target->queue.memcpy(k_num_collision_cells,
                                          this->h_num_collision_cells.data(),
                                          this->cell_count * sizeof(int)));

  auto d_counts =
      get_resource<BufferDevice<INT>, ResourceStackInterfaceBufferDevice<INT>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
          sycl_target);
  d_counts->realloc_no_copy(this->cell_count);
  INT *k_counts = d_counts->ptr;

  es.wait();

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  this->sycl_target->queue
      .parallel_for(
          sycl::nd_range<2>(sycl::range<2>(this->cell_count, local_size),
                            sycl::range<2>(1, local_size)),
          [=](sycl::nd_item<2> ix) {
            const std::size_t mesh_cell = ix.get_global_id(0);
            const std::size_t local_id = ix.get_global_id(1);
            const int num_collision_cells = k_num_collision_cells[mesh_cell];

            INT pair_count_local = 0;
            for (std::size_t cx = local_id; cx < num_collision_cells;
                 cx += local_size) {
              pair_count_local +=
                  k_pair_counts_ccell[mesh_cell * max_num_collision_cells + cx];
            }

            const INT pair_count_mesh_cell = sycl::reduce_over_group(
                ix.get_group(), pair_count_local, sycl::plus<INT>{});

            if (local_id == 0) {
              k_counts[mesh_cell] = pair_count_mesh_cell;
            }
          })
      .wait_and_throw();

  auto k_pair_counts_es = this->d_pair_counts_es->ptr;
  auto k_pair_counts = this->d_pair_counts->ptr;

  auto e1 = this->sycl_target->queue.parallel_for(
      sycl::range<1>(this->cell_count),
      [=](auto ix) { k_pair_counts[ix] = static_cast<int>(k_counts[ix]); });
  auto e3 =
      this->sycl_target->queue.memcpy(this->h_pair_counts.data(), k_pair_counts,
                                      this->cell_count * sizeof(int), e1);

  e1.wait_and_throw();
  e3.wait_and_throw();

  auto e0 = joint_exclusive_scan(this->sycl_target, this->cell_count, k_counts,
                                 k_pair_counts_es);
  auto e2 = this->sycl_target->queue.memcpy(this->h_pair_counts_es.data(),
                                            k_pair_counts_es,
                                            this->cell_count * sizeof(INT), e0);

  auto e4 = joint_exclusive_scan_n(this->sycl_target, this->cell_count,
                                   max_num_collision_cells, k_pair_counts_ccell,
                                   k_pair_counts_ccell_es);

  for (int cx = 0; cx < this->cell_count; cx++) {
    const auto current_size = static_cast<int>(this->d_pair_list->nrow[cx]);
    const auto required_size = this->h_pair_counts[cx];
    if (required_size > current_size) {
      this->d_pair_list->set_nrow(cx, required_size);
    }
  }
  this->d_pair_list->wait_set_nrow();

  e0.wait_and_throw();
  e2.wait_and_throw();
  e4.wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<INT>{}, d_counts);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_pair_counts_ccell_es);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_pair_counts_ccell);
}

CellwisePairListDevice PairSamplerNoReplacement::get_pair_list() {}

CellwisePairListHostMap PairSamplerNoReplacement::get_host_pair_list() {}

} // namespace NESO::Particles::DSMC
