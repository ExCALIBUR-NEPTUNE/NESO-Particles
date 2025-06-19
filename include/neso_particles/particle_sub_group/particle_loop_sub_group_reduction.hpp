#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_REDUCTION_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_REDUCTION_HPP_

#include "particle_loop_sub_group.hpp"

namespace NESO::Particles {

/**
 * Derived ParticleLoop type which implements the particle loop over iteration
 * sets defined by ParticleSubGroups.
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoopSubGroupReduction
    : public ParticleLoopSubGroup<KERNEL, ARGS...> {

protected:
  using typename ParticleLoop<KERNEL, ARGS...>::loop_parameter_type;
  using typename ParticleLoop<KERNEL, ARGS...>::kernel_parameter_type;
  using ParticleLoopArgs<ARGS...>::create_loop_args;
  using ParticleLoopArgs<ARGS...>::create_kernel_args;
  using ParticleLoopArgs<ARGS...>::reduction_initialise_dispatch;
  using ParticleLoopArgs<ARGS...>::reduction_finalise_dispatch;

public:
  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleSubGroup.
   *
   *  @param name Identifier for particle loop.
   *  @param particle_group ParticleSubGroup to execute kernel for all
   * particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopSubGroupReduction(const std::string name,
                                ParticleSubGroupSharedPtr particle_sub_group,
                                KERNEL kernel, ARGS... args)
      : ParticleLoopSubGroup<KERNEL, ARGS...>(name, particle_sub_group, kernel,
                                              args...) {
    this->loop_type = "ParticleLoopSubGroupReduction";
  }

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleSubGroup.
   *
   *  @param particle_group ParticleSubGroup to execute kernel for all
   * particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopSubGroupReduction(ParticleSubGroupSharedPtr particle_sub_group,
                                KERNEL kernel, ARGS... args)
      : ParticleLoopSubGroupReduction("unnamed_kernel", particle_sub_group,
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

    ParticleSubGroupImplementation::MapLoopLayerToLayer
        k_map_cells_to_particles;
    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;

    if (!this->prepare_submit(k_map_cells_to_particles, global_info, cell_start,
                              cell_end)) {
      return;
    }

    auto k_kernel = ParticleLoopImplementation::get_kernel(this->kernel);

    for (auto &blockx : this->iteration_set->iteration_set) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            create_loop_args(cgh, loop_args, &global_info);
            cgh.parallel_for<>(
                blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
                  std::size_t loop_cell;
                  std::size_t loop_layer;
                  block_device.get_cell_layer(idx, &loop_cell, &loop_layer);
                  const int loop_cellx = static_cast<int>(loop_cell);
                  const int loop_layerx = static_cast<int>(loop_layer);
                  ParticleLoopImplementation::ParticleLoopIteration iterationx;
                  iterationx.local_sycl_index = idx.get_local_id(1);
                  iterationx.local_sycl_range = idx.get_local_range(1);
                  iterationx.cellx = loop_cellx;
                  iterationx.loop_layerx = loop_layerx;

                  reduction_initialise_dispatch(idx, iterationx, loop_args);
                  idx.barrier(sycl::access::fence_space::local_space);

                  if (block_device.work_item_required(loop_cell, loop_layer)) {
                    const int layer = static_cast<int>(
                        k_map_cells_to_particles.map_loop_layer_to_layer(
                            loop_cell, loop_layer));
                    iterationx.layerx = layer;
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
