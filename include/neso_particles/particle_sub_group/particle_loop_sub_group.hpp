#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_LOOP_SUB_GROUP_HPP_

#include "../loop/particle_loop.hpp"
#include "particle_sub_group_base.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {

/**
 * Derived ParticleLoop type which implements the particle loop over iteration
 * sets defined by ParticleSubGroups.
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoopSubGroup : public ParticleLoop<KERNEL, ARGS...> {

protected:
  ParticleSubGroupSharedPtr particle_sub_group;

  using typename ParticleLoop<KERNEL, ARGS...>::loop_parameter_type;
  using typename ParticleLoop<KERNEL, ARGS...>::kernel_parameter_type;
  using ParticleLoop<KERNEL, ARGS...>::create_loop_args;
  using ParticleLoop<KERNEL, ARGS...>::create_kernel_args;

  virtual inline int get_loop_type_int() override { return 1; }

  inline void setup_subgroup_is(
      ParticleSubGroupImplementation::SubGroupSelectorBase::SelectionT
          &selection) {
    this->h_npart_cell_lb = selection.h_npart_cell;
    this->d_npart_cell_lb = selection.d_npart_cell;
    this->d_npart_cell_es_lb = selection.d_npart_cell_es;
    this->iteration_set = std::make_unique<
        ParticleLoopImplementation::ParticleLoopBlockIterationSet>(
        this->sycl_target, selection.ncell, this->h_npart_cell_lb,
        this->d_npart_cell_lb);
  }

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
  ParticleLoopSubGroup(const std::string name,
                       ParticleSubGroupSharedPtr particle_sub_group,
                       KERNEL kernel, ARGS... args)
      : ParticleLoop<KERNEL, ARGS...>(name, particle_sub_group->particle_group,
                                      kernel, args...),
        particle_sub_group(particle_sub_group) {
    this->loop_type = "ParticleLoopSubGroup";
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
  ParticleLoopSubGroup(ParticleSubGroupSharedPtr particle_sub_group,
                       KERNEL kernel, ARGS... args)
      : ParticleLoopSubGroup("unnamed_kernel", particle_sub_group, kernel,
                             args...) {}

  /**
   *  Launch the ParticleLoop and return. Must be called collectively over the
   *  MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   *
   *  @param cell Argument for api compatibility.
   */
  inline void submit(const std::optional<int> cell = std::nullopt) override {
    auto t0 = profile_timestamp();
    this->profiling_region_init();

    NESOASSERT(
        (!this->loop_running) || (cell != std::nullopt),
        "ParticleLoop::submit called - but the loop is already submitted.");

    // If the loop is called cell wise asynchronously then the call over cell i
    // could trigger a rebuild on cell i+1
    if (!this->loop_running) {
      this->particle_sub_group->create_if_required();
    }
    this->loop_running = true;

    if (this->iteration_set_is_empty(cell)) {
      return;
    }

    auto &selection = this->particle_sub_group->selection;
    this->setup_subgroup_is(selection);
    auto global_info = this->create_global_info(cell);
    global_info.particle_sub_group = this->particle_sub_group.get();
    this->apply_pre_loop(global_info);

    auto k_kernel = ParticleLoopImplementation::get_kernel(this->kernel);
    auto k_map_cells_to_particles = selection.d_map_cells_to_particles;

    this->profiling_region_metrics(this->iteration_set->iteration_set_size);

    auto &is = this->iteration_set->iteration_set;
    if (cell != std::nullopt) {
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_single_cell(cell.value(),
                                                global_info.local_size, 0);
    } else {
      const std::size_t nbin = this->sycl_target->parameters
                                   ->template get<SizeTParameter>("LOOP_NBIN")
                                   ->value;
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_all_cells(nbin, global_info.local_size, 0);
    }
    this->sycl_target->profile_map.inc(
        "ParticleLoopSubGroup", "Init", 1,
        profile_elapsed(t0, profile_timestamp()));

    for (auto &blockx : is) {
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
                  if (block_device.work_item_required(loop_cell, loop_layer)) {
                    const int layer = static_cast<int>(
                        k_map_cells_to_particles[loop_cell][0][loop_layer]);
                    iterationx.local_sycl_index = idx.get_local_id(1);
                    iterationx.cellx = loop_cellx;
                    iterationx.layerx = layer;
                    iterationx.loop_layerx = loop_layerx;
                    kernel_parameter_type kernel_args;
                    create_kernel_args(iterationx, loop_args, kernel_args);
                    Tuple::apply(k_kernel, kernel_args);
                  }
                });
          }));
    }
  }
};

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param particle_group ParticleSubGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleSubGroupSharedPtr particle_group, KERNEL kernel,
              ARGS... args) {
  if (particle_group->is_entire_particle_group()) {
    return particle_loop(particle_group->get_particle_group(), kernel, args...);
  } else {
    auto p = std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
        particle_group, kernel, args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  }
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param name Identifier for particle loop.
 *  @param particle_group ParticleSubGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name, ParticleSubGroupSharedPtr particle_group,
              KERNEL kernel, ARGS... args) {
  if (particle_group->is_entire_particle_group()) {
    return particle_loop(name, particle_group->get_particle_group(), kernel,
                         args...);
  } else {
    auto p = std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
        name, particle_group, kernel, args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  }
}

} // namespace NESO::Particles

#endif
