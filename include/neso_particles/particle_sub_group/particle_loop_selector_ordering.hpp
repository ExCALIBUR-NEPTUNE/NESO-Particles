#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SELECTOR_ORDERING_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SELECTOR_ORDERING_HPP_

#include "../loop/particle_loop_reduction.hpp"

namespace NESO::Particles {

/**
 * Derived ParticleLoop type which implements the particle loop over iteration
 * sets defined by ParticleSubGroups.
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoopSelectorOrdering
    : public ParticleLoopReduction<KERNEL, ARGS...> {

protected:
  /// The types of the parameters for the outside loops.
  using loop_parameter_type =
      typename ParticleLoopArgs<ARGS...>::loop_parameter_type;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type =
      typename ParticleLoopArgs<ARGS...>::kernel_parameter_type;
  using ParticleLoopArgs<ARGS...>::create_loop_args;
  using ParticleLoopArgs<ARGS...>::create_kernel_args;
  using ParticleLoopArgs<ARGS...>::reduction_initialise_dispatch;
  using ParticleLoopArgs<ARGS...>::reduction_finalise_dispatch;

  int **hd_cell_counts;
  int **hd_layers;

public:
  /// Disable (implicit) copies.
  ParticleLoopSelectorOrdering(const ParticleLoopSelectorOrdering &st) = delete;
  /// Disable (implicit) copies.
  ParticleLoopSelectorOrdering &
  operator=(ParticleLoopSelectorOrdering const &a) = delete;
  virtual ~ParticleLoopSelectorOrdering() = default;

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleGroup.
   *
   *  @param hd_cell_counts Host pointer to device pointer for cellwise particle
   * counts.
   *  @param hd_layers Host pointer to device pointer for particle layers.
   *  @param particle_group ParticleGroup to execute kernel for all particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopSelectorOrdering(int **hd_cell_counts, int **hd_layers,
                               ParticleGroupSharedPtr particle_group,
                               KERNEL kernel, ARGS... args)
      : ParticleLoopReduction<KERNEL, ARGS...>(
            "unnamed_selector_ordering", particle_group, kernel, args...) {
    this->hd_cell_counts = hd_cell_counts;
    this->hd_layers = hd_layers;
    this->local_nbytes_group = sizeof(int);
    this->local_nbytes_item = sizeof(int) * (NESO_PARTICLES_LOOP_STRIDE + 1);
  }

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleGroup. This is a specisalised constructor for a ParticleSubGroup
   * which is actually a whole ParticleGroup.
   *
   *  @param hd_cell_counts Host pointer to device pointer for cellwise particle
   * counts.
   *  @param hd_layers Host pointer to device pointer for particle layers.
   *  @param particle_group ParticleGroup to execute kernel for all particles.
   *  @param particle_sub_group ParticleSubGroup which wraps the entire
   * ParticleGroup.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopSelectorOrdering(
      int **hd_cell_counts, int **hd_layers,
      ParticleGroupSharedPtr particle_group,
      std::shared_ptr<ParticleSubGroup> particle_sub_group, KERNEL kernel,
      ARGS... args)
      : ParticleLoopReduction<KERNEL, ARGS...>(
            "unnamed_selector_ordering", particle_group, particle_sub_group,
            kernel, args...) {
    this->hd_cell_counts = hd_cell_counts;
    this->hd_layers = hd_layers;
    this->local_nbytes_group = sizeof(int);
    this->local_nbytes_item = sizeof(int) * (NESO_PARTICLES_LOOP_STRIDE + 1);
  }

  /**
   *  Launch the ParticleLoop and return. Must be called collectively over the
   *  MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   *
   *  submit() Launches the ParticleLoop over all cells.
   *  submit(i) Launches the ParticleLoop over cell i.
   *  submit(i, i+4) Launches the ParticleLoop over cells i, i+1, i+2, i+3.
   *  Note cell_end itself is not visited.
   *
   *  @param cell_start Optional starting cell to launch the ParticleLoop over.
   *  @param cell_end Optional ending cell to launch the ParticleLoop over.
   */
  virtual inline void
  submit(const std::optional<int> cell_start = std::nullopt,
         const std::optional<int> cell_end = std::nullopt) override {

    int *k_cell_counts = *this->hd_cell_counts;
    int *k_layers = *this->hd_layers;
    auto cell_count = this->particle_group_ptr->domain->mesh->get_cell_count();
    auto e0 = this->sycl_target->queue.fill(k_cell_counts, static_cast<int>(0),
                                            cell_count);

    this->iteration_set_stride = NESO_PARTICLES_LOOP_STRIDE;

    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;

    if (!this->prepare_submit(global_info, cell_start, cell_end)) {
      return;
    }

    auto k_kernel = ParticleLoopImplementation::get_kernel(this->kernel);

    e0.wait_and_throw();
    for (auto &blockx : this->iteration_set->iteration_set) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(this->sycl_target->queue.submit([&](sycl::handler
                                                                     &cgh) {
        loop_parameter_type loop_args;
        ParticleLoopArgs<ARGS...>::create_loop_args(cgh, loop_args,
                                                    &global_info);

        sycl::local_accessor<int, 1> group_offset(sycl::range<1>(1), cgh);
        sycl::local_accessor<int, 1> item_masks(
            sycl::range<1>(global_info.local_size * NESO_PARTICLES_LOOP_STRIDE),
            cgh);
        // sycl::local_accessor<int, 1> item_totals(
        //     sycl::range<1>(global_info.local_size), cgh);

        auto particle_loop_index = ParticleLoopIndex{};
        auto particle_loop_index_accessor = Access::read(&particle_loop_index);
        ParticleLoopImplementation::ParticleLoopIndexKernelT
            particle_loop_index_loop_arg = create_loop_arg(
                &global_info, cgh, particle_loop_index_accessor);

        cgh.parallel_for<>(blockx.loop_iteration_set, [=](sycl::nd_item<2>
                                                              idx) {
          std::size_t cell;
          std::size_t layer;
          ParticleLoopImplementation::ParticleLoopIteration iterationx;

          ParticleLoopImplementation::ParticleLoopIndexKernelT
              particle_loop_index_loop_arg_inner = particle_loop_index_loop_arg;

          const std::size_t local_sycl_index = idx.get_local_id(1);
          const std::size_t local_sycl_range = idx.get_local_range(1);
          iterationx.local_sycl_index = local_sycl_index;
          iterationx.local_sycl_range = local_sycl_range;

          int item_total = 0;

          for (std::size_t stridex = 0; stridex < NESO_PARTICLES_LOOP_STRIDE;
               stridex++) {
            block_device.get_interlaced_cell_layer(idx, stridex, &cell, &layer);

            const std::size_t cellx = cell;
            const std::size_t layerx = layer;

            iterationx.cellx = static_cast<int>(cellx);
            iterationx.layerx = static_cast<int>(layerx);
            iterationx.loop_layerx = static_cast<int>(layerx);

            int mask = 0;
            if (block_device.work_item_required(cellx, layerx)) {
              kernel_parameter_type kernel_args;
              create_kernel_args(iterationx, loop_args, kernel_args);
              mask = static_cast<int>(
                  static_cast<bool>(Tuple::apply(k_kernel, kernel_args)));
            }
            item_total += mask;
            item_masks[local_sycl_index + stridex * local_sycl_range] = mask;
          }
          idx.barrier(sycl::access::fence_space::local_space);

          int ex_item_total = exclusive_scan_over_group(
              idx.get_group(), item_total, sycl::plus<int>());

          // get the offset for the particles in this cell
          if (local_sycl_index == (local_sycl_range - 1)) {
            group_offset[0] = atomic_fetch_add(&k_cell_counts[cell],
                                               ex_item_total + item_total);
          }
          idx.barrier(sycl::access::fence_space::local_space);

          // each work item now can compute a base offset for the
          // particles it will revisit
          item_total = group_offset[0] + ex_item_total;

          // loop back over the particles and assign them their layer if
          // the mask is 1 otherwise -1
          for (std::size_t stridex = 0; stridex < NESO_PARTICLES_LOOP_STRIDE;
               stridex++) {
            block_device.get_interlaced_cell_layer(idx, stridex, &cell, &layer);
            const std::size_t cellx = cell;
            const std::size_t layerx = layer;
            iterationx.cellx = static_cast<int>(cellx);
            iterationx.layerx = static_cast<int>(layerx);
            iterationx.loop_layerx = static_cast<int>(layerx);

            const int mask =
                item_masks[local_sycl_index + stridex * local_sycl_range];
            const int layer = mask ? item_total : -1;
            item_total += mask;

            if (block_device.work_item_required(cellx, layerx)) {
              Access::LoopIndex::Read loop_index;
              ParticleLoopImplementation::create_kernel_arg(
                  iterationx, particle_loop_index_loop_arg_inner, loop_index);
              const INT particle_linear_index =
                  loop_index.get_local_linear_index();
              k_layers[particle_linear_index] = layer;
            }
          }
        });
      }));
    }
  }
};

} // namespace NESO::Particles

#endif
