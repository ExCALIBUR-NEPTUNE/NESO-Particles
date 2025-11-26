#ifndef __NESO_PARTICLES_PAIR_LOOP_PARTICLE_PAIR_LOOP_CELLWISE_PAIR_LIST_HPP_
#define __NESO_PARTICLES_PAIR_LOOP_PARTICLE_PAIR_LOOP_CELLWISE_PAIR_LIST_HPP_

#include "../particle_sub_group/particle_sub_group_utility.hpp"
#include "cellwise_pair_list_absolute.hpp"
#include "particle_pair_loop_args.hpp"

namespace NESO::Particles {

/**
 * ParticlePairLoop from cellwise pair lists. Assumes all pairs are mutually
 * exclusive.
 */
template <typename KERNEL, typename... ARGS>
class ParticlePairLoopCellwisePairList : public ParticlePairLoopArgs<ARGS...> {

protected:
  /// The types of the parameters for the outside loops.
  using loop_parameter_type =
      typename ParticlePairLoopArgs<ARGS...>::loop_parameter_type;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type =
      typename ParticlePairLoopArgs<ARGS...>::kernel_parameter_type;

  using KernelMasksType =
      typename ParticlePairLoopArgs<ARGS...>::KernelMasksType;

  using ParticlePairLoopArgs<ARGS...>::create_loop_args;
  using ParticlePairLoopArgs<ARGS...>::create_kernel_args;

  std::vector<CellwisePairListDevice> h_pair_lists_device;
  std::shared_ptr<BufferDevice<CellwisePairListDevice>> d_pair_lists_device;
  std::size_t num_pair_lists{0};
  int cell_count{0};

  std::vector<INT> h_pair_list_counts_es;
  std::shared_ptr<BufferDevice<INT>> d_pair_list_counts_es;

  virtual inline std::size_t get_iteration_set_size() override {
    std::size_t num_pairs = 0;
    for (const auto &ix : this->h_pair_lists_device) {
      num_pairs += static_cast<std::size_t>(ix.pair_count);
    }
    return num_pairs;
  }

public:
  std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists;
  typename GetParticlePairLoopKernelType<KERNEL>::type kernel;

  /**
   * Create a ParticlePairLoop from cellwise pair lists.
   *
   * @param name Name for ParticlePairLoop.
   * @param pair_lists Vector of CellwisePairListAbsolute pair lists.
   * @param kernel Kernel for pair loop.
   * @param args... Arguments for pair loop.
   */
  ParticlePairLoopCellwisePairList(
      std::string name,
      std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists,
      KERNEL kernel, ARGS... args)
      : ParticlePairLoopArgs<ARGS...>(nullptr, name, nullptr, nullptr, nullptr,
                                      nullptr, args...),
        pair_lists(pair_lists), kernel(particle_pair_loop_kernel(kernel)) {

    if (pair_lists.size()) {
      this->num_pair_lists = pair_lists.size();
      this->particle_group_A = get_particle_group(pair_lists[0].A);
      this->particle_group_B = get_particle_group(pair_lists[0].B);
      NESOASSERT(this->particle_group_A->domain ==
                     this->particle_group_B->domain,
                 "Domain missmatch between ParticleGroups.");

      NESOASSERT(this->particle_group_A->sycl_target ==
                     this->particle_group_B->sycl_target,
                 "Compute device missmatch between ParticleGroups.");

      this->sycl_target = this->particle_group_A->sycl_target;
      this->cell_count = this->particle_group_A->domain->mesh->get_cell_count();
      NESOASSERT(this->sycl_target != nullptr, "Bad compute device found");
      this->d_pair_lists_device =
          std::make_shared<BufferDevice<CellwisePairListDevice>>(
              this->sycl_target, pair_lists.size());
      this->h_pair_lists_device.resize(pair_lists.size());
      this->h_pair_list_counts_es.resize(pair_lists.size());
      this->d_pair_list_counts_es = std::make_shared<BufferDevice<INT>>(
          this->sycl_target, pair_lists.size());
    }
  }

  virtual inline void
  execute([[maybe_unused]] const std::optional<int> cell_start = std::nullopt,
          [[maybe_unused]] const std::optional<int> cell_end =
              std::nullopt) override {
    this->submit(cell_start, cell_end);
    this->wait();
  }

  virtual inline void
  submit([[maybe_unused]] const std::optional<int> cell_start = std::nullopt,
         [[maybe_unused]] const std::optional<int> cell_end =
             std::nullopt) override {

    int max_pair_count = 0;
    INT pair_list_counts_es = 0;
    for (std::size_t listx = 0; listx < this->num_pair_lists; listx++) {
      this->h_pair_lists_device[listx] =
          this->pair_lists[listx].pair_list->get();
      max_pair_count = std::max(
          max_pair_count, this->h_pair_lists_device[listx].max_pair_count);
      // Assemble the exclusive counts across the lists.
      this->h_pair_list_counts_es[listx] = pair_list_counts_es;
      pair_list_counts_es += this->h_pair_lists_device[listx].pair_count;
    }
    this->event_stack.push(
        this->d_pair_lists_device->set_async(this->h_pair_lists_device));
    this->event_stack.push(
        this->d_pair_list_counts_es->set_async(this->h_pair_list_counts_es));

    // Create global info needs to be called after h_pair_lists_device is set.
    this->create_global_info(cell_start, cell_end, &this->global_info_A,
                             &this->global_info_B);

    KernelMasksType kernel_masks;
    this->apply_pre_loop(kernel_masks, this->global_info_A,
                         this->global_info_B);

    const auto cell_start_actual = this->global_info_A.starting_cell;
    const auto cell_end_actual = this->global_info_A.bounding_cell;
    const int cell_count_iteration = cell_end_actual - cell_start_actual;

    this->event_stack.wait();
    if ((max_pair_count > 0) && (this->num_pair_lists > 0) &&
        (cell_count_iteration > 0)) {

      auto k_kernel = this->kernel.kernel;
      auto k_pair_lists = this->d_pair_lists_device->ptr;
      auto k_pair_list_counts_es = this->d_pair_list_counts_es->ptr;
      std::size_t local_size = this->global_info_A.local_size;

      sycl::range<3> local_iteration_set(1, 1, local_size);
      sycl::range<3> global_iteration_set(
          static_cast<std::size_t>(cell_count_iteration),
          static_cast<std::size_t>(this->num_pair_lists),
          static_cast<std::size_t>(local_size));

      sycl::nd_range<3> nd_iteration_set =
          this->sycl_target->device_limits.validate_nd_range(
              sycl::nd_range<3>(global_iteration_set, local_iteration_set));

      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            KernelMasksType kernel_masks;
            this->create_loop_args(cgh, loop_args, kernel_masks,
                                   &this->global_info_A, &this->global_info_B);

            cgh.parallel_for(nd_iteration_set, [=](sycl::nd_item<3> idx) {
              const std::size_t index_cell =
                  idx.get_global_id(0) + cell_start_actual;
              const std::size_t index_list = idx.get_global_id(1);
              const std::size_t local_id = idx.get_local_linear_id();
              const std::size_t local_range =
                  idx.get_group().get_local_linear_range();
              const INT offset_list = k_pair_list_counts_es[index_list];

              const auto *pair_list = &k_pair_lists[index_list];
              const int wave_count = pair_list->get_num_waves(index_cell);

              for (int wavex = 0; wavex < wave_count; wavex++) {

                const auto num_pairs =
                    static_cast<std::size_t>(pair_list->get_num_pairs(
                        wavex, static_cast<int>(index_cell)));

                for (std::size_t index_pair = local_id; index_pair < num_pairs;
                     index_pair += local_range) {

                  const INT offset_cell = pair_list->get_pair_linear_index(
                      wavex, index_cell, index_pair);

                  ParticlePairLoopImplementation::ParticlePairLoopIteration
                      iteration;
                  iteration.work_item = &idx;

                  ParticleLoopImplementation::ParticleLoopIteration iteration_A;
                  ParticleLoopImplementation::ParticleLoopIteration iteration_B;

                  const int particle_index_a = pair_list->get_particle_index_i(
                      wavex, index_cell, index_pair);
                  const int particle_index_b = pair_list->get_particle_index_j(
                      wavex, index_cell, index_pair);

                  iteration.pair_index = offset_list + offset_cell;

                  iteration_A.local_sycl_index = idx.get_local_linear_id();
                  iteration_A.local_sycl_range =
                      idx.get_group().get_local_linear_range();
                  iteration_A.cellx = index_cell;
                  iteration_A.layerx = particle_index_a;

                  iteration_B.local_sycl_index = idx.get_local_linear_id();
                  iteration_B.local_sycl_range =
                      idx.get_group().get_local_linear_range();
                  iteration_B.cellx = index_cell;
                  iteration_B.layerx = particle_index_b;

                  kernel_parameter_type kernel_args;
                  KernelMasksType kernel_masks;

                  create_kernel_args(iteration, kernel_masks, iteration_A,
                                     iteration_B, loop_args, kernel_args);
                  Tuple::apply(k_kernel, kernel_args);
                }

                sycl::group_barrier(idx.get_group());
              }
            });
          }));
    }
  }
};

/**
 * Create a ParticlePairLoop from cellwise pair lists. This is a helper function
 * to create instances of ParticlePairLoopCellwisePairList.
 *
 * @param name Name for ParticlePairLoop.
 * @param pair_lists Vector of CellwisePairListAbsolute pair lists.
 * @param kernel Kernel for pair loop.
 * @param args... Arguments for pair loop.
 * @returns ParticlePairLoop for iteration set and arguments.
 */
template <typename KERNEL, typename... ARGS>
inline ParticlePairLoopBaseSharedPtr particle_pair_loop(
    std::string name,
    std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists,
    KERNEL kernel, ARGS... args) {
  auto ptr =
      std::make_shared<ParticlePairLoopCellwisePairList<KERNEL, ARGS...>>(
          name, pair_lists, kernel, args...);
  return std::dynamic_pointer_cast<ParticlePairLoopBase>(ptr);
}

/**
 * Create a ParticlePairLoop from cellwise pair lists. This is a helper function
 * to create instances of ParticlePairLoopCellwisePairList.
 *
 * @param pair_lists Vector of CellwisePairListAbsolute pair lists.
 * @param kernel Kernel for pair loop.
 * @param args... Arguments for pair loop.
 * @returns ParticlePairLoop for iteration set and arguments.
 */
template <typename KERNEL, typename... ARGS>
inline ParticlePairLoopBaseSharedPtr particle_pair_loop(
    std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists,
    KERNEL kernel, ARGS... args) {

  return particle_pair_loop("unnamed_particle_pair_loop", pair_lists, kernel,
                            args...);
}

} // namespace NESO::Particles

#endif
