#ifndef _NESO_PARTICLES_LOOP_REDUCTIONS_REDUCTION_LOOP_CELLWISE_BINS_HPP_
#define _NESO_PARTICLES_LOOP_REDUCTIONS_REDUCTION_LOOP_CELLWISE_BINS_HPP_

#include "../particle_loop_args.hpp"
#include "reduction_context_cellwise_bins.hpp"

namespace NESO::Particles {

/**
 *  ParticleLoop loop type. The particle loop applies the given kernel to all
 *  particles in a ParticleGroup. The kernel must be independent of the
 *  execution order (i.e. parallel and unsequenced in C++ terminology).
 *
 *  This ParticleLoop implementation is a specialisation for cellwise reductions
 * described by a ReductionLoopCellwiseBins.
 */
template <typename KERNEL, typename... ARGS>
class ReductionLoopCellwiseBins : public ParticleLoopBase,
                                  public ParticleLoopArgs<ARGS...> {
protected:
  KERNEL kernel;
  /// The types of the parameters for the outside loops.
  using loop_parameter_type =
      typename ParticleLoopArgs<ARGS...>::loop_parameter_type;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type =
      typename ParticleLoopArgs<ARGS...>::kernel_parameter_type;
  using ParticleLoopArgs<ARGS...>::create_loop_args;
  using ParticleLoopArgs<ARGS...>::create_kernel_args;

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

  ReductionContextCellwiseBinsSharedPtr reduction_context;
  sycl::nd_range<3> nd_range{sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)};
  std::size_t cell_offset;

  virtual inline int get_loop_type_int() override {
    return this->particle_sub_group_shrptr ? 1 : 0;
  }

public:
  /// Disable (implicit) copies.
  ReductionLoopCellwiseBins(const ReductionLoopCellwiseBins &st) = delete;
  /// Disable (implicit) copies.
  ReductionLoopCellwiseBins &
  operator=(ReductionLoopCellwiseBins const &a) = delete;
  virtual ~ReductionLoopCellwiseBins() = default;

  /**
   * Create a ParticleLoop that executes a kernel for all particles in the
   * reduction context.
   *
   * @param name Identifier for particle loop.
   * @param reduction_context Cellwise bins reduction context for particle loop.
   * @param kernel Kernel to execute for all particles in the ParticleGroup.
   * @param args The remaining arguments are arguments to be passed to the
   *             kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ReductionLoopCellwiseBins(
      const std::string name,
      ReductionContextCellwiseBinsSharedPtr reduction_context, KERNEL kernel,
      ARGS... args)
      : ParticleLoopBase(name, reduction_context->particle_group),
        ParticleLoopArgs<ARGS...>(args...), kernel(kernel),
        reduction_context(reduction_context) {
    this->sycl_target = reduction_context->particle_group->sycl_target;
    this->particle_group_ptr = this->particle_group_shrptr.get();
    this->particle_sub_group_shrptr = reduction_context->particle_sub_group;
    this->loop_type = "ReductionLoopCellwiseBins";
    this->ncell =
        reduction_context->particle_group->domain->mesh->get_cell_count();
    this->init_from_particle_dat(
        reduction_context->particle_group->position_dat);
  };

  virtual inline ParticleLoopImplementation::ParticleLoopGlobalInfo
  create_global_info(
      const std::optional<int> cell_start = std::nullopt,
      const std::optional<int> cell_end = std::nullopt) override {

    int cell_start_v = -1;
    int cell_end_v = -1;
    const bool all_cells = determine_iteration_set(
        this->ncell, cell_start, cell_end, &cell_start_v, &cell_end_v);

    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;
    global_info.particle_group = this->particle_group_ptr;
    global_info.particle_sub_group = this->particle_sub_group_shrptr.get();

    this->d_npart_cell_lb = this->d_npart_cell;
    this->d_npart_cell_es_lb = this->d_npart_cell_es;

    if (this->particle_sub_group_shrptr) {
      this->particle_sub_group_shrptr->create_if_required();
      auto selection = this->particle_sub_group_shrptr->get_selection();
      this->h_npart_cell_lb = selection.h_npart_cell;
      this->d_npart_cell_lb = selection.d_npart_cell;
      this->d_npart_cell_es_lb = selection.d_npart_cell_es;
    }

    global_info.d_npart_cell_lb = this->d_npart_cell_lb;
    global_info.d_npart_cell_es = this->d_npart_cell_es;
    global_info.d_npart_cell_es_lb = this->d_npart_cell_es_lb;
    global_info.all_cells = all_cells;
    global_info.starting_cell = cell_start_v;
    global_info.bounding_cell = cell_end_v;
    global_info.loop_type_int = this->get_loop_type_int();
    global_info.local_size = this->get_local_size();

    return global_info;
  }

protected:
  inline bool prepare_submit(
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info,
      const std::optional<int> cell_start = std::nullopt,
      const std::optional<int> cell_end = std::nullopt) {

    this->profiling_region_init();

    NESOASSERT((!this->loop_running) || (cell_start != std::nullopt),
               "ReductionLoopCellwiseBins::submit called - but the loop is "
               "already submitted.");
    this->loop_running = true;

    int cell_start_v = -1;
    int cell_end_v = -1;
    determine_iteration_set(this->ncell, cell_start, cell_end, &cell_start_v,
                            &cell_end_v);

    global_info = this->create_global_info(cell_start, cell_end);
    this->apply_pre_loop(global_info);

    // This early exit is after the pre loop calls as other ranks may have a
    // non-empty iteration set and collective setup operations in the pre loop.
    if (this->iteration_set_is_empty(cell_start, cell_end)) {
      return false;
    }

    const std::size_t local_size = global_info.local_size;
    const std::size_t stride0 = cell_end_v - cell_start_v;
    const std::size_t stride1 =
        this->reduction_context->partition->get_device().key_strides[1];

    NESOASSERT(
        this->ncell ==
            this->reduction_context->partition->get_device().key_strides[0],
        "Missmatch in map sizes for reduction.");

    this->nd_range =
        sycl::nd_range<3>(sycl::range<3>(stride0, stride1, local_size),
                          sycl::range<3>(1, 1, local_size));

    this->cell_offset = cell_start_v;

    this->profiling_region_metrics(this->iteration_set->iteration_set_size);
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
    auto k_cell_offset = this->cell_offset;
    auto k_nd_range = this->nd_range;
    auto k_index_map = this->reduction_context->partition->get_device();

    this->event_stack.push(
        this->sycl_target->queue.submit([&](sycl::handler &cgh) {
          loop_parameter_type loop_args;
          create_loop_args(cgh, loop_args, &global_info);
          cgh.parallel_for<>(k_nd_range, [=](sycl::nd_item<3> idx) {
            const std::size_t cell = idx.get_global_id(0) + k_cell_offset;
            const std::size_t bin = idx.get_global_id(1);
            const int binx = static_cast<int>(bin);
            const int cellx = static_cast<int>(cell);
            const int key[2] = {cellx, binx};
            const int num_layers = k_index_map.get_num_values(key);
            ParticleLoopImplementation::ParticleLoopIteration iterationx;

            const std::size_t local_sycl_index = idx.get_local_id(2);
            const std::size_t local_sycl_range = idx.get_local_range(2);
            iterationx.local_sycl_index = local_sycl_index;
            iterationx.local_sycl_range = local_sycl_range;

            for (int loop_layerx = local_sycl_index; loop_layerx < num_layers;
                 loop_layerx += local_sycl_range) {
              const int layerx = k_index_map.at(key, loop_layerx, 0);

              iterationx.cellx = cellx;
              iterationx.layerx = layerx;
              iterationx.loop_layerx = layerx;

              kernel_parameter_type kernel_args;
              create_kernel_args(iterationx, loop_args, kernel_args);
              Tuple::apply(k_kernel, kernel_args);
            }
          });
        }));
  }

  /**
   * Wait for loop execution to complete. On completion perform post-loop
   * actions. Must be called collectively on communicator.
   */
  virtual inline void wait() override {
    NESOASSERT(this->loop_running, "ReductionLoopCellwiseBins::wait called - "
                                   "but the loop is not submitted.");
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

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * reduction context. When CellDatConst objects are passed with a Reduction
 * access descriptor then row wise reductions are performed using the reduction
 * context bins.
 *
 *  @param name Identifier for particle loop.
 *  @param reduction_context Cellwise bins reduction context for particle loop.
 *  @param kernel Kernel to execute for all particles in the ParticleGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name,
              ReductionContextCellwiseBinsSharedPtr reduction_context,
              KERNEL kernel, ARGS... args) {
  auto p = std::make_shared<ReductionLoopCellwiseBins<KERNEL, ARGS...>>(
      name, reduction_context, kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * reduction context. When CellDatConst objects are passed with a Reduction
 * access descriptor then row wise reductions are performed using the reduction
 * context bins.
 *
 *  @param reduction_context Cellwise bins reduction context for particle loop.
 *  @param kernel Kernel to execute for all particles in the ParticleGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ReductionContextCellwiseBinsSharedPtr reduction_context,
              KERNEL kernel, ARGS... args) {
  return particle_loop("unnamed_kernel", reduction_context, kernel, args...);
}

} // namespace NESO::Particles

#endif
