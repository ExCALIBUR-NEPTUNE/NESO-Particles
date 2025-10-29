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

  inline void
  setup_subgroup_is(ParticleSubGroupImplementation::Selection &selection) {
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

protected:
  inline bool prepare_submit(
      ParticleSubGroupImplementation::MapLoopLayerToLayer
          &k_map_cells_to_particles,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info,
      const std::optional<int> cell_start = std::nullopt,
      const std::optional<int> cell_end = std::nullopt) {
    this->profiling_region_init();
    auto pr = ProfileRegion(this->loop_type, "prepare_submit");

    // If the loop is called cell wise asynchronously then the call over cell i
    // could trigger a rebuild on cell i+1
    NESOASSERT(
        (!this->loop_running) || (cell_start != std::nullopt),
        "ParticleLoop::submit called - but the loop is already submitted.");

    // If the loop is called cell wise asynchronously then the call over cell i
    // could trigger a rebuild on cell i+1

    auto pr2 = ProfileRegion(this->loop_type, "create_if_required");
    if (!this->loop_running) {
      this->particle_sub_group->create_if_required();
    }
    pr2.end();
    this->sycl_target->profile_map.add_region(pr2);
    this->loop_running = true;

    int cell_start_v, cell_end_v;
    const bool all_cells = determine_iteration_set(
        this->ncell, cell_start, cell_end, &cell_start_v, &cell_end_v);

    auto selection = this->particle_sub_group->get_selection();
    this->setup_subgroup_is(selection);
    k_map_cells_to_particles = selection.d_map_cells_to_particles;

    global_info = this->create_global_info(cell_start, cell_end);
    global_info.particle_sub_group = this->particle_sub_group.get();

    auto pr1 = ProfileRegion(this->loop_type, "apply_pre_loop");
    this->apply_pre_loop(global_info);
    pr1.end();
    this->sycl_target->profile_map.add_region(pr1);

    // This early exit is after the pre loop calls as other ranks may have a
    // non-empty iteration set and collective setup operations in the pre loop.
    if (this->iteration_set_is_empty(cell_start, cell_end)) {
      return false;
    }

    this->profiling_region_metrics(this->iteration_set->iteration_set_size);

    auto &is = this->iteration_set->iteration_set;
    if (all_cells) {
      const std::size_t nbin = this->sycl_target->parameters
                                   ->template get<SizeTParameter>("LOOP_NBIN")
                                   ->value;
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_all_cells(nbin, global_info.local_size, 0,
                                              this->iteration_set_stride);
    } else {
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_range_cell(cell_start_v, cell_end_v,
                                               global_info.local_size, 0,
                                               this->iteration_set_stride);
    }

    pr.end();
    this->sycl_target->profile_map.add_region(pr);
    return true;
  }

public:
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

      auto lambda_dispatch = [&]() {
        this->event_stack.push(
            this->sycl_target->queue.submit([&](sycl::handler &cgh) {
              loop_parameter_type loop_args;
              create_loop_args(cgh, loop_args, &global_info);
              cgh.parallel_for<>(blockx.loop_iteration_set, [=](sycl::nd_item<2>
                                                                    idx) {
                std::size_t loop_cell;
                std::size_t loop_layer;
                block_device.get_cell_layer(idx, &loop_cell, &loop_layer);
                const int loop_cellx = static_cast<int>(loop_cell);
                const int loop_layerx = static_cast<int>(loop_layer);
                ParticleLoopImplementation::ParticleLoopIteration iterationx;
                if (block_device.work_item_required(loop_cell, loop_layer)) {
                  const int layer = static_cast<int>(
                      k_map_cells_to_particles.map_loop_layer_to_layer(
                          loop_cell, loop_layer));
                  iterationx.local_sycl_index = idx.get_local_id(1);
                  iterationx.local_sycl_range = idx.get_local_range(1);
                  iterationx.cellx = loop_cellx;
                  iterationx.layerx = layer;
                  iterationx.loop_layerx = loop_layerx;
                  kernel_parameter_type kernel_args;
                  create_kernel_args(iterationx, loop_args, kernel_args);
                  Tuple::apply(k_kernel, kernel_args);
                }
              });
            }));
      };

#ifdef NESO_PARTICLES_SINGLE_COMPILED_LOOP
      lambda_dispatch();
#else
      if (blockx.layer_bounds_check_required) {
        lambda_dispatch();
      } else {
        this->event_stack.push(
            this->sycl_target->queue.submit([&](sycl::handler &cgh) {
              loop_parameter_type loop_args;
              create_loop_args(cgh, loop_args, &global_info);
              cgh.parallel_for<>(blockx.loop_iteration_set, [=](sycl::nd_item<2>
                                                                    idx) {
                std::size_t loop_cell;
                std::size_t loop_layer;
                block_device.get_cell_layer(idx, &loop_cell, &loop_layer);
                const int loop_cellx = static_cast<int>(loop_cell);
                const int loop_layerx = static_cast<int>(loop_layer);
                ParticleLoopImplementation::ParticleLoopIteration iterationx;
                const int layer = static_cast<int>(
                    k_map_cells_to_particles.map_loop_layer_to_layer(
                        loop_cell, loop_layer));
                iterationx.local_sycl_index = idx.get_local_id(1);
                iterationx.local_sycl_range = idx.get_local_range(1);
                iterationx.cellx = loop_cellx;
                iterationx.layerx = layer;
                iterationx.loop_layerx = loop_layerx;
                kernel_parameter_type kernel_args;
                create_kernel_args(iterationx, loop_args, kernel_args);
                Tuple::apply(k_kernel, kernel_args);
              });
            }));
      }
#endif
    }
  }
};

} // namespace NESO::Particles

#endif
