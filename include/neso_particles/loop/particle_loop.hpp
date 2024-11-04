#ifndef _NESO_PARTICLES_PARTICLE_LOOP_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_H_

#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "../cell_dat.hpp"
#include "../compute_target.hpp"
#include "../containers/descendant_products.hpp"
#include "../containers/global_array.hpp"
#include "../containers/local_array.hpp"
#include "../containers/local_memory_block.hpp"
#include "../containers/nd_local_array.hpp"
#include "../containers/product_matrix.hpp"
#include "../containers/rng/kernel_rng.hpp"
#include "../containers/sym_vector.hpp"
#include "../containers/tuple.hpp"
#include "../particle_dat.hpp"
#include "../particle_spec.hpp"
#include "../sycl_typedefs.hpp"
#include "kernel.hpp"
#include "particle_loop_base.hpp"
#include "particle_loop_index.hpp"
#include "particle_loop_iteration_set.hpp"
#include "pli_particle_dat.hpp"

namespace NESO::Particles {

namespace ParticleLoopImplementation {
/**
 * Catch all for args passed as shared ptrs
 */
template <template <typename> typename T, typename U, typename V>
struct LoopParameter<T<std::shared_ptr<U>>, V> {
  using type = typename LoopParameter<T<U>, V>::type;
};
/**
 * Catch all for args passed as shared ptrs
 */
template <template <typename> typename T, typename U, typename V>
struct KernelParameter<T<std::shared_ptr<U>>, V> {
  using type = typename KernelParameter<T<U>, V>::type;
};

} // namespace ParticleLoopImplementation

namespace {

/**
 *  This is a metafunction which is passed a input data type and returns the
 *  LoopParamter type that corresponds to the input type (by using the structs
 * defined above).
 */
template <class T>
using loop_parameter_t =
    typename ParticleLoopImplementation::LoopParameter<T, std::true_type>::type;

/**
 *  Function to map access descriptor and data structure type to kernel type.
 */
template <class T>
using kernel_parameter_t =
    typename ParticleLoopImplementation::KernelParameter<T,
                                                         std::true_type>::type;

} // namespace

/**
 *  ParticleLoop loop type. The particle loop applies the given kernel to all
 *  particles in a ParticleGroup. The kernel must be independent of the
 *  execution order (i.e. parallel and unsequenced in C++ terminology).
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoop : public ParticleLoopBase {

protected:
  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  static inline auto create_loop_arg_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, T<std::shared_ptr<U>> a) {
    T<U *> c = {a.obj.get()};
    return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
  }
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  static inline auto create_loop_arg_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, T<U> a) {
    T<U *> c = {&a.obj};
    return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
  }

  /*
   * -----------------------------------------------------------------
   */

  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  static inline auto post_loop_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      T<std::shared_ptr<U>> a) {
    T<U *> c = {a.obj.get()};
    ParticleLoopImplementation::post_loop(global_info, c);
  }
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  static inline auto post_loop_cast(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info, T<U> a) {
    T<U *> c = {&a.obj};
    ParticleLoopImplementation::post_loop(global_info, c);
  }
  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  static inline auto
  pre_loop_cast(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                T<std::shared_ptr<U>> a) {
    T<U *> c = {a.obj.get()};
    ParticleLoopImplementation::pre_loop(global_info, c);
  }
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  static inline auto
  pre_loop_cast(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                T<U> a) {
    T<U *> c = {&a.obj};
    ParticleLoopImplementation::pre_loop(global_info, c);
  }

  /*
   * =================================================================
   */

  /// The types of the parameters for the outside loops.
  using loop_parameter_type = Tuple::Tuple<loop_parameter_t<ARGS>...>;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type = Tuple::Tuple<kernel_parameter_t<ARGS>...>;
  /// Tuple of the arguments passed to the ParticleLoop on construction.
  std::tuple<ARGS...> args;

  /// Recursively assemble the tuple args.
  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->args) = a0;
  }
  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->args) = a0;
    this->unpack_args<INDEX + 1>(args...);
  }

  /// Recursively assemble the outer loop arguments.
  template <size_t INDEX, size_t SIZE, typename PARAM>
  inline void create_loop_args_inner(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, PARAM &loop_args) {
    if constexpr (INDEX < SIZE) {
      Tuple::get<INDEX>(loop_args) =
          create_loop_arg_cast(global_info, cgh, std::get<INDEX>(this->args));
      create_loop_args_inner<INDEX + 1, SIZE>(global_info, cgh, loop_args);
    }
  }
  inline void create_loop_args(
      sycl::handler &cgh, loop_parameter_type &loop_args,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {
    create_loop_args_inner<0, sizeof...(ARGS)>(global_info, cgh, loop_args);
  }

  /// recusively assemble the kernel arguments from the loop arguments
  template <size_t INDEX, size_t SIZE>
  static inline constexpr void create_kernel_args_inner(
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
  static inline constexpr void create_kernel_args(
      ParticleLoopImplementation::ParticleLoopIteration &iterationx,
      const loop_parameter_type &loop_args,
      kernel_parameter_type &kernel_args) {

    create_kernel_args_inner<0, sizeof...(ARGS)>(iterationx, loop_args,
                                                 kernel_args);
  }

  ParticleGroupSharedPtr particle_group_shrptr;
  ParticleGroup *particle_group_ptr = {nullptr};
  SYCLTargetSharedPtr sycl_target;
  /// This stores the particle dat the loop was created with to prevent use
  /// after free errors in the case when the ParticleLoop is created with a
  /// ParticleDat.
  std::shared_ptr<void> particle_dat_init;
  KERNEL kernel;
  std::unique_ptr<ParticleLoopImplementation::ParticleLoopBlockIterationSet>
      iteration_set;
  std::string loop_type;
  std::string name;
  EventStack event_stack;
  bool loop_running = {false};
  // The actual number of particles in the cell
  int *d_npart_cell;
  int *h_npart_cell_lb;
  // The number of particles in the cell from the loop bounds point of view.
  int *d_npart_cell_lb;
  // Exclusive sum of the actual number of particles in the cell.
  INT *d_npart_cell_es;
  // Exclusive sum of the number of particles in the cell from the loop bounds
  // point of view.
  INT *d_npart_cell_es_lb;
  int ncell;

  template <typename T>
  inline void init_from_particle_dat(ParticleDatSharedPtr<T> particle_dat) {
    this->ncell = particle_dat->ncell;
    this->h_npart_cell_lb = particle_dat->h_npart_cell;
    this->d_npart_cell = particle_dat->d_npart_cell;
    this->d_npart_cell_lb = this->d_npart_cell;
    this->d_npart_cell_es = particle_dat->get_d_npart_cell_es();
    this->d_npart_cell_es_lb = this->d_npart_cell_es;
    this->iteration_set = std::make_unique<
        ParticleLoopImplementation::ParticleLoopBlockIterationSet>(
        particle_dat);
  }

  template <template <typename> typename T, typename U>
  inline void check_is_sym_inner(T<U> arg) {
    static_assert(
        std::is_same<T<U>, Sym<U>>::value == false,
        "Sym based arguments cannot be passed to ParticleLoop with a "
        "ParticleDat iterator. Pass the ParticleDatSharedPtr instead.");
  }
  template <typename T>
  inline void check_is_sym_inner([[maybe_unused]] T arg) {}
  template <template <typename> typename T, typename U>
  inline void check_is_sym_outer(T<U> arg) {
    check_is_sym_inner(arg.obj);
  }

  virtual inline int get_loop_type_int() { return 0; }

  inline std::size_t get_local_size() {

    // Loop over the args and add how many local bytes they each require.
    std::size_t num_bytes = 0;
    auto lambda_size_add = [&](auto argx) {
      num_bytes +=
          ParticleLoopImplementation::get_required_local_num_bytes(argx);
    };
    auto lambda_size = [&](auto... as) { (lambda_size_add(as), ...); };
    std::apply(lambda_size, this->args);

    // The amount of local space on the device and required number of local
    // bytes gives an upper bound on local size.
    std::size_t local_size =
        this->sycl_target->parameters
            ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    local_size =
        this->sycl_target->get_num_local_work_items(num_bytes, local_size);

    this->sycl_target->profile_map.set("ParticleLoop::" + this->name,
                                       "local_size", local_size, 0.0);

    return local_size;
  }

  inline ParticleLoopImplementation::ParticleLoopGlobalInfo
  create_global_info(const std::optional<int> cell = std::nullopt) {
    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;
    global_info.particle_group = this->particle_group_ptr;
    global_info.particle_sub_group = nullptr;
    global_info.d_npart_cell_es = this->d_npart_cell_es;
    global_info.d_npart_cell_es_lb = this->d_npart_cell_es_lb;

    global_info.starting_cell = (cell == std::nullopt) ? 0 : cell.value();
    global_info.bounding_cell =
        (cell == std::nullopt) ? this->ncell : cell.value() + 1;

    global_info.loop_type_int = this->get_loop_type_int();
    global_info.local_size = this->get_local_size();
    return global_info;
  }

  inline bool iteration_set_is_empty(const std::optional<int> cell) {
    if (cell != std::nullopt) {
      const int cellx = cell.value();
      NESOASSERT(
          (cell > -1) && (cell < this->ncell),
          "ParticleLoop execute or submit called on cell that does not exist.");
      return this->h_npart_cell_lb[cellx] == 0;
    } else if (this->particle_group_ptr != nullptr) {
      return this->particle_group_ptr->get_npart_local() == 0;
    } else {
      return false;
    }
  }

  inline void apply_pre_loop(
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info) {
    auto cast_wrapper = [&](auto t) { pre_loop_cast(&global_info, t); };
    auto pre_loop_caller = [&](auto... as) { (cast_wrapper(as), ...); };
    std::apply(pre_loop_caller, this->args);
  }

  ProfileRegion profile_region;

  inline void profiling_region_init() {
    this->profile_region = ProfileRegion(this->loop_type, this->name);
  }

  inline void profiling_region_metrics(const std::size_t size) {
    this->profile_region.num_bytes =
        size * ParticleLoopImplementation::get_kernel_num_bytes(this->kernel);
    this->profile_region.num_flops =
        size * ParticleLoopImplementation::get_kernel_num_flops(this->kernel);
  }

  inline void profile_region_finalise() {
    this->profile_region.end();
    this->sycl_target->profile_map.add_region(this->profile_region);
  }

public:
  /// Disable (implicit) copies.
  ParticleLoop(const ParticleLoop &st) = delete;
  /// Disable (implicit) copies.
  ParticleLoop &operator=(ParticleLoop const &a) = delete;

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
      : name(name), particle_group_shrptr(particle_group), kernel(kernel) {

    this->sycl_target = particle_group->sycl_target;
    this->particle_group_ptr = this->particle_group_shrptr.get();
    this->loop_type = "ParticleLoop";
    this->init_from_particle_dat(particle_group->position_dat);
    this->unpack_args<0>(args...);
  };

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
      : ParticleLoop("unnamed_kernel", particle_group, kernel, args...){};

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
      : name(name), kernel(kernel) {

    this->sycl_target = particle_dat->sycl_target;
    this->particle_group_shrptr = nullptr;
    this->particle_group_ptr = nullptr;
    this->loop_type = "ParticleLoop";
    this->init_from_particle_dat(particle_dat);
    this->particle_dat_init = std::static_pointer_cast<void>(particle_dat);
    this->unpack_args<0>(args...);
    (check_is_sym_outer(args), ...);
  };

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
      : ParticleLoop("unnamed_kernel", particle_dat, kernel, args...){};

  /**
   *  Launch the ParticleLoop and return. Must be called collectively over the
   *  MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   *
   *  @param cell Optional cell index to only launch the ParticleLoop over.
   */
  inline void submit(const std::optional<int> cell = std::nullopt) {
    this->profiling_region_init();

    NESOASSERT(
        (!this->loop_running) || (cell != std::nullopt),
        "ParticleLoop::submit called - but the loop is already submitted.");
    this->loop_running = true;
    if (this->iteration_set_is_empty(cell)) {
      return;
    }

    auto t0 = profile_timestamp();
    auto global_info = this->create_global_info(cell);
    this->apply_pre_loop(global_info);

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
    this->profiling_region_metrics(this->iteration_set->iteration_set_size);
    auto k_kernel = ParticleLoopImplementation::get_kernel(this->kernel);
    this->sycl_target->profile_map.inc(
        "ParticleLoop", "Init", 1, profile_elapsed(t0, profile_timestamp()));

    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(sycl_target->queue.submit([&](sycl::handler &cgh) {
        loop_parameter_type loop_args;
        create_loop_args(cgh, loop_args, &global_info);
        cgh.parallel_for<>(
            blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
              std::size_t cell;
              std::size_t layer;
              block_device.get_cell_layer(idx, &cell, &layer);
              const int cellx = static_cast<int>(cell);
              const int layerx = static_cast<int>(layer);
              ParticleLoopImplementation::ParticleLoopIteration iterationx;
              if (block_device.work_item_required(cell, layer)) {
                iterationx.local_sycl_index = idx.get_local_id(1);
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
  }

  /**
   * Wait for loop execution to complete. On completion perform post-loop
   * actions. Must be called collectively on communicator.
   */
  inline void wait() {
    NESOASSERT(this->loop_running,
               "ParticleLoop::wait called - but the loop is not submitted.");
    // wait for the loop execution to complete
    this->event_stack.wait();
    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info =
        this->create_global_info();
    auto cast_wrapper = [&](auto t) { post_loop_cast(&global_info, t); };
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
   *  @param cell Optional cell to launch the ParticleLoop over.
   */
  inline void execute(const std::optional<int> cell = std::nullopt) {
    auto t0 = profile_timestamp();
    this->submit(cell);
    this->wait();
    this->sycl_target->profile_map.inc(
        this->loop_type, this->name, 1,
        profile_elapsed(t0, profile_timestamp()));
  }
};

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
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleGroupSharedPtr particle_group, KERNEL kernel,
              ARGS... args) {
  auto p = std::make_shared<ParticleLoop<KERNEL, ARGS...>>(particle_group,
                                                           kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

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
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name, ParticleGroupSharedPtr particle_group,
              KERNEL kernel, ARGS... args) {
  auto p = std::make_shared<ParticleLoop<KERNEL, ARGS...>>(name, particle_group,
                                                           kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
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
template <typename DAT_TYPE, typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleDatSharedPtr<DAT_TYPE> particle_dat, KERNEL kernel,
              ARGS... args) {
  auto p = std::make_shared<ParticleLoop<KERNEL, ARGS...>>(particle_dat, kernel,
                                                           args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

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
template <typename DAT_TYPE, typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name,
              ParticleDatSharedPtr<DAT_TYPE> particle_dat, KERNEL kernel,
              ARGS... args) {
  auto p = std::make_shared<ParticleLoop<KERNEL, ARGS...>>(name, particle_dat,
                                                           kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

} // namespace NESO::Particles

#endif
