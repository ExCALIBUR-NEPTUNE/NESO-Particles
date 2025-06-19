#ifndef _NESO_PARTICLES_LOOP_PARTICLE_LOOP_REDUCTION_H_
#define _NESO_PARTICLES_LOOP_PARTICLE_LOOP_REDUCTION_H_

#include "particle_loop.hpp"

namespace NESO::Particles {

/**
 *  ParticleLoop loop type for when the loop contains a reduction.
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoopReduction : public ParticleLoop<KERNEL, ARGS...> {
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

public:
  /// Disable (implicit) copies.
  ParticleLoopReduction(const ParticleLoopReduction &st) = delete;
  /// Disable (implicit) copies.
  ParticleLoopReduction &operator=(ParticleLoopReduction const &a) = delete;
  virtual ~ParticleLoopReduction() = default;

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleGroup.
   *
   *  @param name Identifier for particle loop.
   *  @param particle_group ParticleGroup to execute kernel for all particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopReduction(const std::string name,
                        ParticleGroupSharedPtr particle_group, KERNEL kernel,
                        ARGS... args)
      : ParticleLoop<KERNEL, ARGS...>(name, particle_group, kernel, args...) {}

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleGroup. This is a specisalised constructor for a ParticleSubGroup
   * which is actually a whole ParticleGroup.
   *
   *  @param name Identifier for particle loop.
   *  @param particle_group ParticleGroup to execute kernel for all particles.
   *  @param particle_sub_group ParticleSubGroup which wraps the entire
   * ParticleGroup.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopReduction(const std::string name,
                        ParticleGroupSharedPtr particle_group,
                        std::shared_ptr<ParticleSubGroup> particle_sub_group,
                        KERNEL kernel, ARGS... args)
      : ParticleLoop<KERNEL, ARGS...>(name, particle_group, particle_sub_group,
                                      kernel, args...) {}

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

    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;

    if (!this->prepare_submit(global_info, cell_start, cell_end)) {
      return;
    }

    auto k_kernel = ParticleLoopImplementation::get_kernel(this->kernel);

    for (auto &blockx : this->iteration_set->iteration_set) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            ParticleLoopArgs<ARGS...>::create_loop_args(cgh, loop_args,
                                                        &global_info);
            cgh.parallel_for<>(
                blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
                  std::size_t cell;
                  std::size_t layer;
                  block_device.get_cell_layer(idx, &cell, &layer);
                  const int cellx = static_cast<int>(cell);
                  const int layerx = static_cast<int>(layer);
                  ParticleLoopImplementation::ParticleLoopIteration iterationx;
                  iterationx.local_sycl_index = idx.get_local_id(1);
                  iterationx.local_sycl_range = idx.get_local_range(1);
                  iterationx.cellx = cellx;
                  iterationx.layerx = layerx;
                  iterationx.loop_layerx = layerx;
                  reduction_initialise_dispatch(idx, iterationx, loop_args);
                  idx.barrier(sycl::access::fence_space::local_space);

                  if (block_device.work_item_required(cell, layer)) {
                    kernel_parameter_type kernel_args;
                    create_kernel_args(iterationx, loop_args, kernel_args);
                    Tuple::apply(k_kernel, kernel_args);
                  }

                  idx.barrier(sycl::access::fence_space::local_space);
                  reduction_finalise_dispatch(idx, iterationx, loop_args);
                });
          }));
    }
  }
};

} // namespace NESO::Particles

#endif
