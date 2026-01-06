#ifndef _NESO_PARTICLES_PAIR_LOOP_PAIR_LOOP_ARGS_HPP_
#define _NESO_PARTICLES_PAIR_LOOP_PAIR_LOOP_ARGS_HPP_

#include "../containers/rng/kernel_rng.hpp"
#include "../loop/particle_loop_args.hpp"
#include "../particle_sub_group/particle_sub_group_base.hpp"
#include "particle_pair_loop_base.hpp"
#include "particle_pair_loop_index.hpp"

namespace NESO::Particles {

/**
 * Base type for handling the arguments for a pair loop.
 */
template <typename... ARGS>
class ParticlePairLoopArgs : public ParticlePairLoopBase {
protected:
  /// The types of the parameters for the outside loops.
  using loop_parameter_type = Tuple::Tuple<loop_parameter_t<
      typename Access::StripPairGroupAnnotation<ARGS>::type>...>;

  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type = Tuple::Tuple<kernel_parameter_t<
      typename Access::StripPairGroupAnnotation<ARGS>::type>...>;

  /// The mask types for the arguments passed to the kernel
  using KernelMasksType =
      Tuple::Tuple<typename Access::GetAnotateMask<ARGS>::mask...>;

  /// Tuple of the arguments passed to the ParticlePairLoop on construction.
  std::tuple<ARGS...> annotated_args;
  /// Tuple of the arguments with the A/B specification stripped away. These
  /// should be the same types that would be passed to ParticleLoop.
  std::tuple<typename Access::StripPairGroupAnnotation<ARGS>::type...> args;

  ParticleLoopImplementation::ParticleLoopGlobalInfo global_info_A;
  ParticleLoopImplementation::ParticleLoopGlobalInfo global_info_B;
  ParticleGroupSharedPtr particle_group_A{nullptr};
  ParticleGroupSharedPtr particle_group_B{nullptr};
  ParticleSubGroupSharedPtr particle_sub_group_A{nullptr};
  ParticleSubGroupSharedPtr particle_sub_group_B{nullptr};

  EventStack event_stack;
  std::optional<ProfileRegion> profile_region;

  /// Recursively assemble the tuple args.
  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->annotated_args) = a0;
    std::get<INDEX>(this->args) = Access::strip_pair_group_annotation(a0);
  }
  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->annotated_args) = a0;
    std::get<INDEX>(this->args) = Access::strip_pair_group_annotation(a0);
    this->unpack_args<INDEX + 1>(args...);
  }

  /**
   * Method to compute access to a type wrapped in a shared_ptr.
   */
  template <template <typename> typename T, typename U>
  inline void
  pre_loop_cast(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                T<std::shared_ptr<U>> a) {
    T<U *> c = {a.obj.get()};
    ParticleLoopImplementation::pre_loop(global_info, c);
  }
  /**
   * Method to compute access to a type not wrapper in a shared_ptr
   */
  template <template <typename> typename T, typename U>
  inline void
  pre_loop_cast(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                T<U> a) {
    T<U *> c = {&a.obj};
    ParticleLoopImplementation::pre_loop(global_info, c);
  }

  template <size_t INDEX, size_t SIZE>
  inline void apply_pre_loop_inner(
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_A,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_B) {

    if constexpr (INDEX < SIZE) {
      if constexpr (Access::IsAnnotatedB<decltype(Tuple::get<INDEX>(
                        kernel_masks_type))>::value) {
        pre_loop_cast(&global_info_B, std::get<INDEX>(this->args));
      } else {
        pre_loop_cast(&global_info_A, std::get<INDEX>(this->args));
      }
      apply_pre_loop_inner<INDEX + 1, SIZE>(kernel_masks_type, global_info_A,
                                            global_info_B);
    }
  }

  inline void apply_pre_loop(
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_A,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_B) {
    apply_pre_loop_inner<0, sizeof...(ARGS)>(kernel_masks_type, global_info_A,
                                             global_info_B);
  }

  template <size_t INDEX, size_t SIZE>
  inline void apply_post_loop_inner(
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_A,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_B) {

    if constexpr (INDEX < SIZE) {
      if constexpr (Access::IsAnnotatedB<decltype(Tuple::get<INDEX>(
                        kernel_masks_type))>::value) {
        post_loop_cast(&global_info_B, std::get<INDEX>(this->args));
      } else {
        post_loop_cast(&global_info_A, std::get<INDEX>(this->args));
      }
      apply_post_loop_inner<INDEX + 1, SIZE>(kernel_masks_type, global_info_A,
                                             global_info_B);
    }
  }

  inline void apply_post_loop(
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_A,
      ParticleLoopImplementation::ParticleLoopGlobalInfo &global_info_B) {
    apply_post_loop_inner<0, sizeof...(ARGS)>(kernel_masks_type, global_info_A,
                                              global_info_B);
  }

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

  /// Recursively assemble the outer loop arguments.
  template <size_t INDEX, size_t SIZE, typename PARAM>
  inline void create_loop_args_inner(
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info_A,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info_B,
      sycl::handler &cgh, PARAM &loop_args) {
    if constexpr (INDEX < SIZE) {

      if constexpr (Access::IsAnnotatedB<decltype(Tuple::get<INDEX>(
                        kernel_masks_type))>::value) {
        Tuple::get<INDEX>(loop_args) = create_loop_arg_cast(
            global_info_B, cgh, std::get<INDEX>(this->args));
      } else {
        Tuple::get<INDEX>(loop_args) = create_loop_arg_cast(
            global_info_A, cgh, std::get<INDEX>(this->args));
      }

      create_loop_args_inner<INDEX + 1, SIZE>(kernel_masks_type, global_info_A,
                                              global_info_B, cgh, loop_args);
    }
  }

  inline void create_loop_args(
      sycl::handler &cgh, loop_parameter_type &loop_args,
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info_A,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info_B

  ) {
    create_loop_args_inner<0, sizeof...(ARGS)>(kernel_masks_type, global_info_A,
                                               global_info_B, cgh, loop_args);
  }

  /// recusively assemble the kernel arguments from the loop arguments
  template <size_t INDEX, size_t SIZE>
  static inline void create_kernel_args_inner(
      ParticlePairLoopImplementation::ParticlePairLoopIteration &iteration,
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_A,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_B,
      const loop_parameter_type &loop_args,
      kernel_parameter_type &kernel_args) {

    if constexpr (INDEX < SIZE) {
      auto arg = Tuple::get<INDEX>(loop_args);

      if constexpr (Access::IsAnnotatedB<decltype(Tuple::get<INDEX>(
                        kernel_masks_type))>::value) {
        ParticlePairLoopImplementation::create_kernel_arg(
            iteration, iteration_B, arg, Tuple::get<INDEX>(kernel_args));
      } else {
        ParticlePairLoopImplementation::create_kernel_arg(
            iteration, iteration_A, arg, Tuple::get<INDEX>(kernel_args));
      }
      create_kernel_args_inner<INDEX + 1, SIZE>(iteration, kernel_masks_type,
                                                iteration_A, iteration_B,
                                                loop_args, kernel_args);
    }
  }

  /// called before kernel execution to assemble the kernel arguments.
  static inline void create_kernel_args(
      ParticlePairLoopImplementation::ParticlePairLoopIteration &iteration,
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_A,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_B,
      const loop_parameter_type &loop_args,
      kernel_parameter_type &kernel_args) {

    create_kernel_args_inner<0, sizeof...(ARGS)>(iteration, kernel_masks_type,
                                                 iteration_A, iteration_B,
                                                 loop_args, kernel_args);
  }

  /**
   * @returns The number of pairs in the iteration set.
   */
  virtual inline std::size_t get_iteration_set_size() = 0;

  inline void create_global_info(
      const std::optional<int> cell_start = std::nullopt,
      const std::optional<int> cell_end = std::nullopt,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info_A =
          nullptr,
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info_B =
          nullptr) {

    int cell_start_actual = 0;
    const int cell_count =
        this->particle_group_A->domain->mesh->get_cell_count();
    int cell_end_actual = cell_count;

    if (cell_start != std::nullopt) {
      cell_start_actual = cell_start.value();
    }
    if (cell_end != std::nullopt) {
      cell_end_actual = cell_end.value();
    }

    const bool default_start = cell_start_actual == 0;
    const bool default_end = cell_end_actual == cell_count;
    const bool all_cells = default_start && default_end;

    const auto local_size =
        this->sycl_target->parameters
            ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;

    const auto iteration_set_size = this->get_iteration_set_size();

    global_info_A->particle_group = this->particle_group_A.get();
    global_info_A->particle_sub_group =
        this->particle_sub_group_A ? this->particle_sub_group_A.get() : nullptr;
    global_info_A->all_cells = all_cells;
    global_info_A->starting_cell = cell_start_actual;
    global_info_A->bounding_cell = cell_end_actual;
    global_info_A->local_size = local_size;
    global_info_A->provided_iteration_set_size = true;
    global_info_A->iteration_set_size = iteration_set_size;

    global_info_B->particle_group = this->particle_group_B.get();
    global_info_B->particle_sub_group =
        this->particle_sub_group_B ? this->particle_sub_group_B.get() : nullptr;
    global_info_B->all_cells = all_cells;
    global_info_B->starting_cell = cell_start_actual;
    global_info_B->bounding_cell = cell_end_actual;
    global_info_B->local_size = local_size;
    global_info_B->provided_iteration_set_size = true;
    global_info_B->iteration_set_size = iteration_set_size;
  }

public:
  SYCLTargetSharedPtr sycl_target{nullptr};
  std::string name;
  virtual ~ParticlePairLoopArgs<ARGS...>() = default;

  ParticlePairLoopArgs<ARGS...>(SYCLTargetSharedPtr sycl_target,
                                std::string &name,
                                ParticleGroupSharedPtr particle_group_A,
                                ParticleGroupSharedPtr particle_group_B,
                                ParticleSubGroupSharedPtr particle_sub_group_A,
                                ParticleSubGroupSharedPtr particle_sub_group_B,
                                ARGS... args)
      : particle_group_A(particle_group_A), particle_group_B(particle_group_B),
        particle_sub_group_A(particle_sub_group_A),
        particle_sub_group_B(particle_sub_group_B), sycl_target(sycl_target),
        name(name) {
    this->unpack_args<0>(args...);
  }

  virtual inline void wait() override {
    this->event_stack.wait();

    KernelMasksType kernel_masks;
    this->apply_post_loop(kernel_masks, this->global_info_A,
                          this->global_info_B);
    this->sycl_target->profile_map.end_region(this->profile_region);
  }
};

} // namespace NESO::Particles

#endif
