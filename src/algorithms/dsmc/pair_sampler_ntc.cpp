#include <neso_particles/algorithms/dsmc/pair_sampler_ntc.hpp>

namespace NESO::Particles::DSMC {

PairSamplerNTC::PairSamplerNTC(
    SYCLTargetSharedPtr sycl_target, const int cell_count,
    std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function)
    : d_wave_counts(
          std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      h_pair_counts(std::vector<int>(cell_count)),
      d_pair_counts(
          std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      h_pair_counts_es(std::vector<INT>(cell_count)),
      d_pair_counts_es(
          std::make_shared<BufferDevice<INT>>(sycl_target, cell_count)),
      d_pair_list(std::make_shared<BufferDevice<int>>(sycl_target, cell_count)),
      sycl_target(sycl_target), cell_count(cell_count),
      rng_generation_function(rng_generation_function) {

  this->block_size = this->sycl_target->parameters
                         ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
                         ->value;
}

void PairSamplerNTC::sample(ParticleSubGroupSharedPtr sub_group_a,
                            ParticleSubGroupSharedPtr sub_group_b,
                            std::vector<int> &new_sample_counts) {

  auto r0 =
      this->sycl_target->profile_map.start_region("PairSamplerNTC", "sample");

  NESOASSERT(static_cast<int>(new_sample_counts.size()) == this->cell_count,
             "new_sample_counts size does not match the cell count.");

  auto particle_group = get_particle_group(sub_group_a);
  NESOASSERT(particle_group == get_particle_group(sub_group_b),
             "Passed particle sub groups are from different parent particle "
             "groups.");
  NESOASSERT(particle_group->domain->mesh->get_cell_count() == this->cell_count,
             "Missmatch in cell counts.");

  const bool a_is_b = sub_group_a == sub_group_b;

  INT exscan = 0;
  int max_num_blocks = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    this->h_pair_counts[cx] = new_sample_counts[cx];
    const auto required_size = this->h_pair_counts[cx];

    if (a_is_b) {
      NESOASSERT(sub_group_a->get_npart_cell(cx) >= 2 || (required_size == 0),
                 "Insufficent particles to sample pairs.");
    } else {
      NESOASSERT(sub_group_a->get_npart_cell(cx) >= 1 || (required_size == 0),
                 "Insufficent particles to sample pairs.");
      NESOASSERT(sub_group_b->get_npart_cell(cx) >= 1 || (required_size == 0),
                 "Insufficent particles to sample pairs.");
    }

    max_num_blocks = std::max(
        max_num_blocks,
        static_cast<int>(get_next_multiple(required_size, block_size)) /
            block_size);
    this->h_pair_counts_es[cx] = exscan;
    exscan += required_size;
  }
  this->pair_count = exscan;
  this->d_pair_list->realloc_no_copy(this->pair_count * 3);
  const auto k_pair_count = this->pair_count;
  this->d_pair_counts->set(this->h_pair_counts);
  this->d_pair_counts_es->set(this->h_pair_counts_es);
  int *k_pair_counts = this->d_pair_counts->ptr;
  INT *k_pair_counts_es = this->d_pair_counts_es->ptr;
  this->d_wave_counts->realloc_no_copy(max_num_blocks * cell_count);
  int *k_wave_counts = this->d_wave_counts->ptr;

  auto d_real_samples = get_resource<BufferDevice<REAL>,
                                     ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_real_samples->realloc_no_copy(2 * k_pair_count);
  REAL *k_real_samples = d_real_samples->ptr;

  this->rng_generation_function->draw_random_samples(
      this->sycl_target, k_real_samples, 2 * k_pair_count, 1024);

  int *k_pair_list = this->d_pair_list->ptr;

  sub_group_a->create_if_required();
  sub_group_b->create_if_required();

  auto selection_a = sub_group_a->get_selection();
  auto selection_b = sub_group_b->get_selection();

  const auto *k_npart_cell = particle_group->mpi_rank_dat->d_npart_cell;
  const auto *h_npart_cell = particle_group->mpi_rank_dat->h_npart_cell;

  int max_npart_cell = 0;
  for (int cellx = 0; cellx < this->cell_count; cellx++) {
    max_npart_cell = std::max(max_npart_cell, h_npart_cell[cellx]);
  }

  const std::size_t required_local_memory_bytes =
      (max_npart_cell + 1) * sizeof(int);

  NESOASSERT(required_local_memory_bytes <=
                 this->sycl_target->device_limits.local_mem_size,
             "Required local memory size is less than available local "
             "memory. Please raise an issue if this is actually a problem.");

  auto nd_iteration_set = this->sycl_target->device_limits.validate_nd_range(
      sycl::nd_range<2>(sycl::range<2>(this->cell_count, this->block_size),
                        sycl::range<2>(1, this->block_size)));

  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int, 1> la_counts(sycl::range(1), cgh);
        sycl::local_accessor<int, 1> la_flags(sycl::range(max_npart_cell), cgh);

        cgh.parallel_for(nd_iteration_set, [=](sycl::nd_item<2> idx) {
          const int cell = idx.get_global_id(0);
          const int local_id = idx.get_group().get_local_linear_id();
          const int local_range = idx.get_group().get_local_linear_range();
          const int npart_cell = k_npart_cell[cell];
          const int npairs = k_pair_counts[cell];
          const int npart_cell_a = selection_a.d_npart_cell[cell];
          const int npart_cell_b = selection_b.d_npart_cell[cell];
          const int sample_range_a = npart_cell_a;
          const int sample_range_b = a_is_b ? npart_cell_a - 1 : npart_cell_b;
          const REAL scale_a = static_cast<REAL>(sample_range_a);
          const REAL scale_b = static_cast<REAL>(sample_range_b);
          const int offset_cell = k_pair_counts_es[cell];
          int block_index = 0;

          if (local_id == 0) {
            la_counts[0] = 0;
          }
          for (int ix = local_id; ix < npart_cell; ix += local_range) {
            la_flags[ix] = 0;
          }

          idx.barrier(sycl::access::fence_space::local_space);

          for (int num_sampled_pairs = 0; num_sampled_pairs < npairs;
               num_sampled_pairs += local_range) {

            int i3 = -1;
            int j3 = -1;
            int wave = -1000000000;
            int order = 0;
            const int ox = offset_cell + num_sampled_pairs + local_id;

            if ((num_sampled_pairs + local_id) < npairs) {

              const int i0 = scale_a * k_real_samples[ox];
              const int j0 = scale_b * k_real_samples[k_pair_count + ox];
              const int i1 = Kernel::min(i0, sample_range_a - 1);
              const int j1 = Kernel::min(j0, sample_range_b - 1);

              const int i2 = i1;
              const int j2 = a_is_b ? (i1 + 1 + j1) % npart_cell_a : j1;

              i3 = selection_a.d_map_cells_to_particles.map_loop_layer_to_layer(
                  cell, i2);
              j3 = selection_b.d_map_cells_to_particles.map_loop_layer_to_layer(
                  cell, j2);

              const int flag_i =
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group>(la_flags[i3])
                      .fetch_add(1);
              const int flag_j =
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group>(la_flags[j3])
                      .fetch_add(1);

              // If both flags are zero then either there are no conflicts or
              // this work item managed to be the first to flag both indices
              // before any other work item.
              if ((flag_i == 0) && (flag_j == 0)) {
                wave = 0;
              } else {
                order = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group>(
                            la_counts[0])
                            .fetch_add(1) +
                        1;
              }
            }

            // resolve dependencies within the block of pairs
            idx.barrier(sycl::access::fence_space::local_space);
            const int num_steps_ordering = la_counts[0];

            if (num_steps_ordering) {

              if (i3 > -1) {
                la_flags[i3] = 0;
                la_flags[j3] = 0;
              }
              idx.barrier(sycl::access::fence_space::local_space);

              for (int orderx = 0; orderx < num_steps_ordering; orderx++) {

                // Only one of the work items holding a conflicting pair should
                // enter this conditional per iteration.
                if ((orderx + 1) == order) {
                  wave = sycl::max(la_flags[i3], la_flags[j3]) + 1;
                  la_flags[i3] = wave;
                  la_flags[j3] = wave;
                  la_counts[0] = sycl::max(wave, la_counts[0]);
                }

                idx.barrier(sycl::access::fence_space::local_space);
              }

              idx.barrier(sycl::access::fence_space::local_space);
            }
            if (wave >= 0) {
              k_pair_list[ox] = i3;
              k_pair_list[k_pair_count * 1 + ox] = j3;
              k_pair_list[k_pair_count * 2 + ox] = wave;
            }

            if (local_id == 0) {
              k_wave_counts[block_index * cell_count + cell] = la_counts[0] + 1;
              la_counts[0] = 0;
            }
            if (i3 > -1) {
              la_flags[i3] = 0;
              la_flags[j3] = 0;
            }

            idx.barrier(sycl::access::fence_space::local_space);
            block_index++;
          }
        });
      })
      .wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_real_samples);

  this->d_pair_list_block_device = {
      this->pair_count,         this->block_size,
      this->cell_count,         this->d_wave_counts->ptr,
      this->d_pair_counts->ptr, this->d_pair_counts_es->ptr,
      this->d_pair_list->ptr};

  this->sycl_target->profile_map.end_region(r0);
}

CellwisePairListBlockDevice PairSamplerNTC::get_pair_list() {
  return this->d_pair_list_block_device;
}

} // namespace NESO::Particles::DSMC
