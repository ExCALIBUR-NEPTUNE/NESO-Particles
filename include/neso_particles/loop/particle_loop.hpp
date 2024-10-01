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
#include "../containers/nd_local_array.hpp"
#include "../containers/product_matrix.hpp"
#include "../containers/rng/kernel_rng.hpp"
#include "../containers/sym_vector.hpp"
#include "../containers/tuple.hpp"
#include "../particle_dat.hpp"
#include "../particle_spec.hpp"
#include "../sycl_typedefs.hpp"
#include "particle_loop_base.hpp"
#include "particle_loop_index.hpp"
#include "pli_particle_dat.hpp"

namespace NESO::Particles::ParticleLoopImplementation {

/**
 * For a set of cells containing particles create several sycl::nd_range
 * instances which cover the iteration space of all particles. This exists to
 * create an iteration set over all particles which is blocked, to reduce the
 * number of kernel launches, and reasonably robust to non-uniform.
 */
struct ParticleLoopIterationSet {

  /// The number of blocks of cells.
  const int nbin;
  /// The number of cells.
  const int ncell;
  /// Host accessible pointer to the number of particles in each cell.
  int *h_npart_cell;
  /// Container to store the sycl::nd_ranges.
  std::vector<sycl::nd_range<2>> iteration_set;
  /// Offsets to add to the cell index to map to the correct cell.
  std::vector<std::size_t> cell_offsets;

  /**
   *  Creates iteration set creator for a given set of cell particle counts.
   *
   *  @param nbin Number of blocks of cells.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   */
  ParticleLoopIterationSet(const int nbin, const int ncell, int *h_npart_cell)
      : nbin(std::min(ncell, nbin)), ncell(ncell), h_npart_cell(h_npart_cell) {
    this->iteration_set.reserve(nbin);
    this->cell_offsets.reserve(nbin);
  }

  /**
   *  Create and return an iteration set which is formed as nbin
   *  sycl::nd_ranges.
   *
   *  @param cell If set iteration set will only cover this cell.
   *  @param local_size Optional size of SYCL work groups.
   *  @returns Tuple containing: Number of bins, sycl::nd_ranges, cell index
   *  offsets.
   */
  inline std::tuple<int, std::vector<sycl::nd_range<2>> &,
                    std::vector<std::size_t> &>
  get(const std::optional<int> cell = std::nullopt,
      const size_t local_size = 256) {

    this->iteration_set.clear();
    this->cell_offsets.clear();

    if (cell == std::nullopt) {
      for (int binx = 0; binx < nbin; binx++) {
        int start, end;
        get_decomp_1d(nbin, ncell, binx, &start, &end);
        const int bin_width = end - start;
        int cell_maxi = 0;
        int cell_avg = 0;
        for (int cellx = start; cellx < end; cellx++) {
          const int cell_occ = h_npart_cell[cellx];
          cell_maxi = std::max(cell_maxi, cell_occ);
          cell_avg += cell_occ;
        }
        cell_avg = (((REAL)cell_avg) / ((REAL)(end - start)));
        const size_t cell_local_size =
            get_min_power_of_two((size_t)cell_avg, local_size);
        const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                      static_cast<long long>(cell_local_size));
        const std::size_t outer_size =
            static_cast<std::size_t>(div_mod.quot +
                                     (div_mod.rem == 0 ? 0 : 1)) *
            cell_local_size;

        if (cell_maxi > 0) {
          this->iteration_set.emplace_back(
              sycl::nd_range<2>(sycl::range<2>(bin_width, outer_size),
                                sycl::range<2>(1, cell_local_size)));
          this->cell_offsets.push_back(static_cast<std::size_t>(start));
        }
      }

      return {this->iteration_set.size(), this->iteration_set,
              this->cell_offsets};
    } else {
      const int cellx = cell.value();
      const size_t cell_maxi = static_cast<size_t>(h_npart_cell[cellx]);
      const size_t cell_local_size =
          get_min_power_of_two((size_t)cell_maxi, local_size);

      const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                    static_cast<long long>(cell_local_size));
      const std::size_t outer_size =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1)) *
          cell_local_size;
      this->iteration_set.emplace_back(sycl::nd_range<2>(
          sycl::range<2>(1, outer_size), sycl::range<2>(1, cell_local_size)));
      this->cell_offsets.push_back(static_cast<std::size_t>(cellx));
      return {1, this->iteration_set, this->cell_offsets};
    }
  }
};

} // namespace NESO::Particles::ParticleLoopImplementation

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

/**
 * Method to compute access to a type wrapped in a shared_ptr.
 */
template <template <typename> typename T, typename U>
inline auto create_loop_arg_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    sycl::handler &cgh, T<std::shared_ptr<U>> a) {
  T<U *> c = {a.obj.get()};
  return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
}
/**
 * Method to compute access to a type not wrapper in a shared_ptr
 */
template <template <typename> typename T, typename U>
inline auto create_loop_arg_cast(
    ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
    sycl::handler &cgh, T<U> a) {
  T<U *> c = {&a.obj};
  return ParticleLoopImplementation::create_loop_arg(global_info, cgh, c);
}

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
          ParticleLoopImplementation::create_loop_arg_cast(
            global_info, cgh, std::get<INDEX>(this->args));
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
  std::unique_ptr<ParticleLoopImplementation::ParticleLoopIterationSet>
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
  size_t local_size;

  template <typename T>
  inline void init_from_particle_dat(ParticleDatSharedPtr<T> particle_dat) {
    this->ncell = particle_dat->ncell;
    this->h_npart_cell_lb = particle_dat->h_npart_cell;
    this->d_npart_cell = particle_dat->d_npart_cell;
    this->d_npart_cell_lb = this->d_npart_cell;
    this->d_npart_cell_es = particle_dat->get_d_npart_cell_es();
    this->d_npart_cell_es_lb = this->d_npart_cell_es;

    int nbin = 16;
    if (const char *env_nbin = std::getenv("NESO_PARTICLES_LOOP_NBIN")) {
      nbin = std::stoi(env_nbin);
    }
    this->local_size = 256;
    if (const char *env_ls = std::getenv("NESO_PARTICLES_LOOP_LOCAL_SIZE")) {
      this->local_size = std::stoi(env_ls);
    }

    this->iteration_set =
        std::make_unique<ParticleLoopImplementation::ParticleLoopIterationSet>(
            nbin, ncell, this->h_npart_cell_lb);
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

    auto k_npart_cell_lb = this->d_npart_cell_lb;
    auto is = this->iteration_set->get(cell, this->local_size);
    auto k_kernel = this->kernel;

    const int nbin = std::get<0>(is);
    this->sycl_target->profile_map.inc(
        "ParticleLoop", "Init", 1, profile_elapsed(t0, profile_timestamp()));

    for (int binx = 0; binx < nbin; binx++) {

      sycl::nd_range<2> ndr = std::get<1>(is).at(binx);
      const size_t cell_offset = std::get<2>(is).at(binx);
      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            create_loop_args(cgh, loop_args, &global_info);
            cgh.parallel_for<>(ndr, [=](sycl::nd_item<2> idx) {
              // const std::size_t index = idx.get_global_linear_id();
              const size_t cellxs = idx.get_global_id(0) + cell_offset;
              const size_t layerxs = idx.get_global_id(1);
              const int cellx = static_cast<int>(cellxs);
              const int layerx = static_cast<int>(layerxs);
              ParticleLoopImplementation::ParticleLoopIteration iterationx;
              if (layerx < k_npart_cell_lb[cellx]) {
                // iterationx.index = index;
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
