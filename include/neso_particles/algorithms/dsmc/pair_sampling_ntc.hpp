#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_ALL_TO_ALL_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_ALL_TO_ALL_HPP_

#include "../../compute_target.hpp"
#include "../../containers/rng/host_rng_common.hpp"
#include "../../particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

namespace DSMC {

/**
 * Interface to sample pairs of particles for NTC style DSMC.
 */
class PairSamplingNTC {
protected:
  std::vector<int> sample_counts;
  std::shared_ptr<BufferDevice<int>> d_pair_list;

public:
  /// Disable (implicit) copies.
  PairSamplingNTC(const PairSamplingNTC &st) = delete;
  /// Disable (implicit) copies.
  PairSamplingNTC &operator=(PairSamplingNTC const &a) = delete;

  SYCLTargetSharedPtr sycl_target{nullptr};
  int cell_count{0};
  std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function{nullptr};

  PairSamplingNTC(
      SYCLTargetSharedPtr sycl_target, const int cell_count,
      std::shared_ptr<RNGGenerationFunction<REAL>> rng_generation_function)
      : sample_counts(std::vector<int>(cell_count)),
        d_pair_list(std::make_shared<BufferDevice<int>>(sycl_target, 16)),
        sycl_target(sycl_target), cell_count(cell_count),
        rng_generation_function(rng_generation_function) {}

  inline void sample(ParticleSubGroupSharedPtr sub_group_a,
                     ParticleSubGroupSharedPtr sub_group_b,
                     std::vector<int> &new_sample_counts) {

    NESOASSERT(new_sample_counts.size() == this->cell_count,
               "new_sample_counts size does not match the cell count.");

    auto particle_group = get_particle_group(sub_group_a);
    NESOASSERT(particle_group == get_particle_group(sub_group_b),
               "Passed particle sub groups are from different parent particle "
               "groups.");
    NESOASSERT(particle_group->domain->mesh->get_cell_count() ==
                   this->cell_count,
               "Missmatch in cell counts.");

    std::copy(new_sample_counts.begin(), new_sample_counts.end(),
              this->sample_counts.begin());

    auto d_sample_counts =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_sample_counts->realloc_no_copy(this->cell_count + 1);
    int *k_sample_counts = d_sample_counts->ptr;
    auto e0 = this->sycl_target->queue.memcpy(k_sample_counts,
                                              this->sample_counts.data(),
                                              this->cell_count * sizeof(int));

    auto d_sample_counts_es =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_sample_counts_es->realloc_no_copy(this->cell_count + 1);
    int *k_sample_counts_es = d_sample_counts_es->ptr;

    e0.wait_and_throw();
    joint_exclusive_scan(this->sycl_target, this->cell_count + 1,
                         k_sample_counts, k_sample_counts_es)
        .wait_and_throw();

    int total_num_pairs = 0;
    this->sycl_target->queue
        .memcpy(&total_num_pairs, k_sample_counts_es + cell_count, sizeof(int))
        .wait_and_throw();

    auto d_real_samples =
        get_resource<BufferDevice<REAL>,
                     ResourceStackInterfaceBufferDevice<REAL>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<REAL>{}, sycl_target);
    d_real_samples->realloc_no_copy(2 * total_num_pairs);
    REAL *k_real_samples = d_real_samples->ptr;

    this->rng_generation_function->draw_random_samples(
        this->sycl_target, k_real_samples, 2 * total_num_pairs, 1024);

    this->d_pair_list->realloc_no_copy(3 * total_num_pairs);
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

    const auto block_size =
        this->sycl_target->parameters
            ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;

    auto nd_iteration_set = this->sycl_target->device_limits.validate_nd_range(
        sycl::nd_range<2>(sycl::range<2>(this->cell_count, block_size),
                          sycl::range<2>(1, block_size)));

    const bool a_is_b = sub_group_a == sub_group_b;

    this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> la_counts(sycl::range(1), cgh);
      sycl::local_accessor<int, 1> la_flags(sycl::range(max_npart_cell), cgh);

      cgh.parallel_for(nd_iteration_set, [=](sycl::nd_item<2> idx) {
        const int cell = idx.get_global_id(0);
        const int local_id = idx.get_group().get_local_linear_id();
        const int local_range = idx.get_group().get_local_linear_range();
        const int npart_cell = k_npart_cell[cell];
        const int npairs = k_sample_counts[cell];
        const int npart_cell_a = selection_a.d_npart_cell[cell];
        const int npart_cell_b = selection_b.d_npart_cell[cell];
        const int sample_range_a = npart_cell_a;
        const int sample_range_b = a_is_b ? npart_cell_a - 1 : npart_cell_b;
        const REAL scale_a = static_cast<REAL>(sample_range_a);
        const REAL scale_b = static_cast<REAL>(sample_range_b);
        const int offset_cell = k_sample_counts_es[cell];

        if (local_id == 0) {
          la_counts[0] = 0;
        }
        for (int ix = local_id; ix < npart_cell; ix += local_range) {
          la_flags[ix] = 0;
        }

        idx.barrier(sycl::access::fence_space::local_space);

        for (int num_sampled_pairs = 0; num_sampled_pairs < npairs;
             num_sampled_pairs += local_range) {

          int i2 = -1;
          int j2 = -1;
          int wave = -1000000000;
          int order = 0;
          const int ox = offset_cell + num_sampled_pairs + local_id;

          if ((num_sampled_pairs + local_id) < npairs) {

            const int i0 = scale_a * k_real_samples[ox];
            const int j0 = scale_b * k_real_samples[total_num_pairs + ox];
            const int i1 = Kernel::min(i0, sample_range_a - 1);
            const int j1 = Kernel::min(j0, sample_range_b - 1);

            i2 = i1;
            j2 = a_is_b ? (i1 + j1) % npart_cell_a : j1;

            const int flag_i =
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group>(la_flags[i2])
                    .fetch_add(1);
            const int flag_j =
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group>(la_flags[j2])
                    .fetch_add(1);
            // If both flags are zero then either there are no conflicts or
            // this work item managed to be the first to flag both indices
            // before any other work item.

            if ((flag_i == 0) && (flag_j == 0)) {
              wave = 0;
            } else {
              order =
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group>(la_counts[0])
                      .fetch_add(1) +
                  1;
            }
          }

          // resolve dependencies within the block of pairs
          idx.barrier(sycl::access::fence_space::local_space);

          const int num_steps_ordering = la_counts[0];

          for (int orderx = 0; orderx < num_steps_ordering; orderx++) {

            TODO SORT OUT ORDER

                idx.barrier(sycl::access::fence_space::local_space);
          }

          idx.barrier(sycl::access::fence_space::local_space);

          if (wave >= 0) {
            k_pair_list[ox] = i2;
            k_pair_list[total_num_pairs + ox] = j2;
            k_pair_list[2 * total_num_pairs + ox] = wave;
          }

          if (local_id == 0) {
            la_counts[0] = 0;
          }
          if (i2 > -1) {
            la_flags[i2] = 0;
            la_flags[j2] = 0;
          }

          idx.barrier(sycl::access::fence_space::local_space);
        }
      });
    });

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<REAL>{}, d_real_samples);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_sample_counts_es);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_sample_counts);
  }
};

} // namespace DSMC
} // namespace NESO::Particles

#endif
