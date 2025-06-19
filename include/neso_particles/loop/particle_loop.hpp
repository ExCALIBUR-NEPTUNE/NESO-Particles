#ifndef _NESO_PARTICLES_LOOP_PARTICLE_LOOP_H_
#define _NESO_PARTICLES_LOOP_PARTICLE_LOOP_H_

#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "particle_loop_args.hpp"

namespace NESO::Particles {

/**
 *  ParticleLoop loop type. The particle loop applies the given kernel to all
 *  particles in a ParticleGroup. The kernel must be independent of the
 *  execution order (i.e. parallel and unsequenced in C++ terminology).
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoop : public ParticleLoopBase, public ParticleLoopArgs<ARGS...> {
protected:
  KERNEL kernel;
  /// The types of the parameters for the outside loops.
  using loop_parameter_type =
      typename ParticleLoopArgs<ARGS...>::loop_parameter_type;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type =
      typename ParticleLoopArgs<ARGS...>::kernel_parameter_type;

  virtual inline std::size_t get_local_size() override {
    return ParticleLoopArgs<ARGS...>::get_local_size_args(this->sycl_target,
                                                          this->name);
  }

  virtual inline void
  profiling_region_metrics(const std::size_t size) override {
    this->profile_region.num_bytes =
        size * ParticleLoopImplementation::get_kernel_num_bytes(this->kernel);
    this->profile_region.num_flops =
        size * ParticleLoopImplementation::get_kernel_num_flops(this->kernel);
  }

  /// recusively assemble the kernel arguments from the loop arguments
  template <size_t INDEX, size_t SIZE>
  static inline void create_kernel_args_inner(
      ParticleLoopImplementation::ParticleLoopIteration &iterationx,
      const loop_parameter_type &loop_args,
      kernel_parameter_type &kernel_args) {

    if constexpr (INDEX < SIZE) {
      auto arg = Tuple::get<INDEX>(loop_args);
      ParticleLoopImplementation::create_kernel_arg(
          iterationx, arg, Tuple::get<INDEX>(kernel_args));
      create_kernel_args_inner<INDEX + 1, SIZE>(iterationx, loop_args,
                                                kernel_args);
    }
  }

  /// called before kernel execution to assemble the kernel arguments.
  static inline void create_kernel_args(
      ParticleLoopImplementation::ParticleLoopIteration &iterationx,
      const loop_parameter_type &loop_args,
      kernel_parameter_type &kernel_args) {

    create_kernel_args_inner<0, sizeof...(ARGS)>(iterationx, loop_args,
                                                 kernel_args);
  }

public:
  /// Disable (implicit) copies.
  ParticleLoop(const ParticleLoop &st) = delete;
  /// Disable (implicit) copies.
  ParticleLoop &operator=(ParticleLoop const &a) = delete;
  virtual ~ParticleLoop() = default;

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
  ParticleLoop(const std::string name, ParticleGroupSharedPtr particle_group,
               KERNEL kernel, ARGS... args)
      : ParticleLoopBase(name, particle_group),
        ParticleLoopArgs<ARGS...>(args...), kernel(kernel) {
    this->sycl_target = particle_group->sycl_target;
    this->particle_group_ptr = this->particle_group_shrptr.get();
    this->loop_type = "ParticleLoop";
    this->init_from_particle_dat(particle_group->position_dat);
  };

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
  ParticleLoop(const std::string name, ParticleGroupSharedPtr particle_group,
               std::shared_ptr<ParticleSubGroup> particle_sub_group,
               KERNEL kernel, ARGS... args)
      : ParticleLoop(name, particle_group, kernel, args...) {
    this->particle_sub_group_shrptr = particle_sub_group;
  }

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleGroup.
   *
   *  @param particle_group ParticleGroup to execute kernel for all particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoop(ParticleGroupSharedPtr particle_group, KERNEL kernel,
               ARGS... args)
      : ParticleLoop("unnamed_kernel", particle_group, kernel, args...) {}

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleDat.
   *
   *  @param name Identifier for particle loop.
   *  @param particle_dat ParticleDat to define the iteration set.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  template <typename DAT_TYPE>
  ParticleLoop(const std::string name,
               ParticleDatSharedPtr<DAT_TYPE> particle_dat, KERNEL kernel,
               ARGS... args)
      : ParticleLoopArgs<ARGS...>(args...), kernel(kernel) {
    this->name = name;
    this->sycl_target = particle_dat->sycl_target;
    this->particle_group_shrptr = nullptr;
    this->particle_group_ptr = nullptr;
    this->loop_type = "ParticleLoop";
    this->init_from_particle_dat(particle_dat);
    this->particle_dat_init = std::static_pointer_cast<void>(particle_dat);
    (check_is_sym_outer(args), ...);
  }

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleDat.
   *
   *  @param particle_dat ParticleDat to define the iteration set.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  template <typename DAT_TYPE>
  ParticleLoop(ParticleDatSharedPtr<DAT_TYPE> particle_dat, KERNEL kernel,
               ARGS... args)
      : ParticleLoop("unnamed_kernel", particle_dat, kernel, args...) {}

protected:
  inline bool prepare_submit(
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info,
      const std::optional<int> cell_start = std::nullopt,
      const std::optional<int> cell_end = std::nullopt) {

    this->profiling_region_init();

    NESOASSERT(
        (!this->loop_running) || (cell_start != std::nullopt),
        "ParticleLoop::submit called - but the loop is already submitted.");
    this->loop_running = true;

    int cell_start_v = -1;
    int cell_end_v = -1;
    const bool all_cells = determine_iteration_set(
        this->ncell, cell_start, cell_end, &cell_start_v, &cell_end_v);

    if (this->iteration_set_is_empty(cell_start, cell_end)) {
      return false;
    }

    auto t0 = profile_timestamp();
    global_info = this->create_global_info(cell_start, cell_end);
    this->apply_pre_loop(global_info);

    auto &is = this->iteration_set->iteration_set;

    if (all_cells) {
      const std::size_t nbin = this->sycl_target->parameters
                                   ->template get<SizeTParameter>("LOOP_NBIN")
                                   ->value;
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_all_cells(nbin, global_info.local_size, 0);
    } else {
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_range_cell(cell_start_v, cell_end_v,
                                               global_info.local_size, 0);
    }

    this->profiling_region_metrics(this->iteration_set->iteration_set_size);
    this->sycl_target->profile_map.inc(
        "ParticleLoop", "Init", 1, profile_elapsed(t0, profile_timestamp()));
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

    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;

    if (!this->prepare_submit(global_info, cell_start, cell_end)) {
      return;
    }

    auto k_kernel = ParticleLoopImplementation::get_kernel(this->kernel);

    for (auto &blockx : this->iteration_set->iteration_set) {
      const auto block_device = blockx.block_device;
      auto lambda_dispatch = [&]() {
        this->event_stack.push(
            this->sycl_target->queue.submit([&](sycl::handler &cgh) {
              loop_parameter_type loop_args;
              ParticleLoopArgs<ARGS...>::create_loop_args(cgh, loop_args,
                                                          &global_info);
              cgh.parallel_for<>(blockx.loop_iteration_set, [=](sycl::nd_item<2>
                                                                    idx) {
                std::size_t cell;
                std::size_t layer;
                block_device.get_cell_layer(idx, &cell, &layer);
                const int cellx = static_cast<int>(cell);
                const int layerx = static_cast<int>(layer);
                ParticleLoopImplementation::ParticleLoopIteration iterationx;
                if (block_device.work_item_required(cell, layer)) {
                  iterationx.local_sycl_index = idx.get_local_id(1);
                  iterationx.local_sycl_range = idx.get_local_range(1);
                  iterationx.cellx = cellx;
                  iterationx.layerx = layerx;
                  iterationx.loop_layerx = layerx;
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
            sycl_target->queue.submit([&](sycl::handler &cgh) {
              loop_parameter_type loop_args;
              ParticleLoopArgs<ARGS...>::create_loop_args(cgh, loop_args,
                                                          &global_info);
              cgh.parallel_for<>(blockx.loop_iteration_set, [=](sycl::nd_item<2>
                                                                    idx) {
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
                kernel_parameter_type kernel_args;
                create_kernel_args(iterationx, loop_args, kernel_args);
                Tuple::apply(k_kernel, kernel_args);
              });
            }));
      }
#endif
    }
  }

  /**
   * Wait for loop execution to complete. On completion perform post-loop
   * actions. Must be called collectively on communicator.
   */
  virtual inline void wait() override {
    NESOASSERT(this->loop_running,
               "ParticleLoop::wait called - but the loop is not submitted.");
    // wait for the loop execution to complete
    this->event_stack.wait();
    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info =
        this->create_global_info();
    auto cast_wrapper = [&](auto t) {
      ParticleLoopArgs<ARGS...>::post_loop_cast(&global_info, t);
    };
    auto post_loop_caller = [&](auto... as) { (cast_wrapper(as), ...); };
    std::apply(post_loop_caller, this->args);
    this->loop_running = false;
    this->profile_region_finalise();
  }

  /**
   *  Execute the ParticleLoop and block until execution is complete. Must be
   *  called collectively on the MPI communicator associated with the
   *  SYCLTarget this loop is over.
   *
   *  execute() Launches the ParticleLoop over all cells.
   *  execute(i) Launches the ParticleLoop over cell i.
   *  execute(i, i+4) Launches the ParticleLoop over cells i, i+1, i+2, i+3.
   *  Note cell_end itself is not visited.
   *
   *  @param cell_start Optional starting cell to launch the ParticleLoop over.
   *  @param cell_end Optional ending cell to launch the ParticleLoop over.
   */
  virtual inline void
  execute(const std::optional<int> cell_start = std::nullopt,
          const std::optional<int> cell_end = std::nullopt) override {
    auto t0 = profile_timestamp();
    this->submit(cell_start, cell_end);
    this->wait();
    this->sycl_target->profile_map.inc(
        this->loop_type, this->name, 1,
        profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO::Particles

#endif
