#include "include/test_neso_particles.hpp"

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
      std::tuple<typename Access::GetAnotateMask<ARGS>::mask...>;

  /// Tuple of the arguments passed to the ParticlePairLoop on construction.
  std::tuple<ARGS...> annotated_args;
  /// Tuple of the arguments with the A/B specification stripped away. These
  /// should be the same types that would be passed to ParticleLoop.
  std::tuple<typename Access::StripPairGroupAnnotation<ARGS>::type...> args;

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
  static inline void create_kernel_args_inner(
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_A,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_B,
      const loop_parameter_type &loop_args,
      kernel_parameter_type &kernel_args) {

    if constexpr (INDEX < SIZE) {
      auto arg = Tuple::get<INDEX>(loop_args);

      if constexpr (Access::IsAnnotatedA<decltype(std::get<INDEX>(
                        kernel_masks_type))>::value) {
        ParticleLoopImplementation::create_kernel_arg(
            iteration_A, arg, Tuple::get<INDEX>(kernel_args));
      } else {
        ParticleLoopImplementation::create_kernel_arg(
            iteration_B, arg, Tuple::get<INDEX>(kernel_args));
      }
      create_kernel_args_inner<INDEX + 1, SIZE>(
          kernel_masks_type, iteration_A, iteration_B, loop_args, kernel_args);
    }
  }

  /// called before kernel execution to assemble the kernel arguments.
  static inline void create_kernel_args(
      KernelMasksType &kernel_masks_type,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_A,
      ParticleLoopImplementation::ParticleLoopIteration &iteration_B,
      const loop_parameter_type &loop_args,
      kernel_parameter_type &kernel_args) {

    create_kernel_args_inner<0, sizeof...(ARGS)>(
        kernel_masks_type, iteration_A, iteration_B, loop_args, kernel_args);
  }

public:
  std::string name;
  virtual ~ParticlePairLoopArgs<ARGS...>() = default;

  ParticlePairLoopArgs<ARGS...>(std::string &name, ARGS... args) : name(name) {
    this->unpack_args<0>(args...);
  }
};

/**
 * Type for defining the A and B sets along with the pair list.
 */
template <typename GROUP_TYPE> struct CellwisePairListAbsolute;
/**
 * Type for defining the A and B sets along with the pair list where both A and
 * B are ParticleGroups.
 */
template <> struct CellwisePairListAbsolute<ParticleGroup> {
  ParticleGroupSharedPtr A;
  ParticleGroupSharedPtr B;
  DSMC::CellwisePairListSharedPtr pair_list;
  CellwisePairListAbsolute<ParticleGroup>() = default;
  ~CellwisePairListAbsolute<ParticleGroup>() = default;

  CellwisePairListAbsolute<ParticleGroup>(
      ParticleGroupSharedPtr A, ParticleGroupSharedPtr B,
      DSMC::CellwisePairListSharedPtr pair_list)
      : A(A), B(B), pair_list(pair_list) {}
};

/**
 * TODO Assumes all pairs are mutually exclusive.
 */
template <typename KERNEL, typename... ARGS>
class ParticlePairLoopCellwisePairList : public ParticlePairLoopArgs<ARGS...> {

protected:
  /// The types of the parameters for the outside loops.
  using loop_parameter_type =
      typename ParticlePairLoopArgs<ARGS...>::loop_parameter_type;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type =
      typename ParticlePairLoopArgs<ARGS...>::kernel_parameter_type;

  using KernelMasksType =
      typename ParticlePairLoopArgs<ARGS...>::KernelMasksType;

  using ParticlePairLoopArgs<ARGS...>::create_loop_args;
  using ParticlePairLoopArgs<ARGS...>::create_kernel_args;

  SYCLTargetSharedPtr sycl_target;
  std::vector<DSMC::CellwisePairListDevice> h_pair_lists_device;
  std::shared_ptr<BufferDevice<DSMC::CellwisePairListDevice>>
      d_pair_lists_device;
  std::size_t num_pair_lists{0};
  int cell_count{0};
  EventStack event_stack;

public:
  std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists;
  ParticlePairLoopKernel<KERNEL> kernel;

  ParticlePairLoopCellwisePairList(
      std::string name,
      std::vector<CellwisePairListAbsolute<ParticleGroup>> pair_lists,
      ParticlePairLoopKernel<KERNEL> kernel, ARGS... args)
      : ParticlePairLoopArgs<ARGS...>(name, args...), pair_lists(pair_lists),
        kernel(kernel) {

    if (pair_lists.size()) {
      this->num_pair_lists = pair_lists.size();
      auto particle_group = get_particle_group(pair_lists[0].A);
      this->sycl_target = particle_group->sycl_target;
      this->cell_count = particle_group->domain->mesh->get_cell_count();
      NESOASSERT(this->sycl_target != nullptr, "Bad compute device found");
      this->d_pair_lists_device =
          std::make_shared<BufferDevice<DSMC::CellwisePairListDevice>>(
              this->sycl_target, pair_lists.size());
      this->h_pair_lists_device.resize(pair_lists.size());
    }
  }

  virtual inline void
  execute([[maybe_unused]] const std::optional<int> cell_start = std::nullopt,
          [[maybe_unused]] const std::optional<int> cell_end =
              std::nullopt) override {
    this->submit(cell_start, cell_end);
    this->wait();
  }

  virtual inline void
  submit([[maybe_unused]] const std::optional<int> cell_start = std::nullopt,
         [[maybe_unused]] const std::optional<int> cell_end =
             std::nullopt) override {

    int max_pair_count = 0;
    for (std::size_t listx = 0; listx < this->num_pair_lists; listx++) {
      this->h_pair_lists_device[listx] =
          this->pair_lists[listx].pair_list->get();
      max_pair_count = std::max(
          max_pair_count, this->h_pair_lists_device[listx].max_pair_count);
    }
    auto e0 = this->d_pair_lists_device->set_async(this->h_pair_lists_device);

    // START OF IMPLEMENTATION TO MOVE INTO A BASE CLASS
    int cell_start_actual = 0;
    int cell_end_actual = this->cell_count - 1;
    if (cell_start != std::nullopt) {
      cell_start_actual = cell_start.value();
    }
    if (cell_end != std::nullopt) {
      cell_end_actual = cell_end.value();
    }
    const int cell_count_iteration = cell_end_actual - cell_start_actual;

    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;

    global_info.particle_group = get_particle_group(pair_lists[0].A).get();
    global_info.particle_sub_group = nullptr;

    // END OF IMPLEMENTATION TO MOVE INTO A BASE CLASS

    e0.wait_and_throw();
    if ((max_pair_count > 0) && (this->num_pair_lists > 0) &&
        (cell_count_iteration > 0)) {

      auto iteration_set =
          this->sycl_target->device_limits.validate_range_global(
              sycl::range<3>(static_cast<std::size_t>(cell_count_iteration),
                             static_cast<std::size_t>(this->num_pair_lists),
                             static_cast<std::size_t>(max_pair_count)));

      auto k_kernel = this->kernel.kernel;
      auto k_pair_lists = this->d_pair_lists_device->ptr;

      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            this->create_loop_args(cgh, loop_args, &global_info);

            cgh.parallel_for(iteration_set, [=](sycl::item<3> idx) {
              const std::size_t index_cell = idx.get_id(0);
              const std::size_t index_list = idx.get_id(1);
              const std::size_t index_pair = idx.get_id(2);
              const auto *pair_list = &k_pair_lists[index_list];
              const auto num_pairs = static_cast<std::size_t>(
                  pair_list->d_pair_counts[index_cell]);

              ParticleLoopImplementation::ParticleLoopIteration iteration_A;
              ParticleLoopImplementation::ParticleLoopIteration iteration_B;

              if (index_pair < num_pairs) {
                const int particle_index_a =
                    pair_list->d_pair_list[index_cell][0][index_pair];
                const int particle_index_b =
                    pair_list->d_pair_list[index_cell][1][index_pair];

                // Now we can create the kernel args. We need to update the
                // kernel arg creation to dispatch for A or B.
                iteration_A.cellx = index_cell;
                iteration_A.layerx = particle_index_a;
                iteration_B.cellx = index_cell;
                iteration_B.layerx = particle_index_b;

                kernel_parameter_type kernel_args;
                KernelMasksType kernel_masks;

                create_kernel_args(kernel_masks, iteration_A, iteration_B,
                                   loop_args, kernel_args);
                Tuple::apply(k_kernel, kernel_args);
              }
            });
          }));
    }
  }

  virtual inline void wait() override { this->event_stack.wait(); }
};

TEST(ParticlePairLoop, base) {

  const int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 32;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NUM_NEIGHBOURS"), 1);

  particle_loop(
      A, [=](auto NN) { NN.at(0) = 0; },
      Access::write(Sym<INT>("NUM_NEIGHBOURS")))
      ->execute();

  auto cellwise_pair_listA =
      std::make_shared<DSMC::CellwisePairList>(sycl_target, cell_count);

  std::vector<int> c = {0};
  std::vector<int> i = {0};
  std::vector<int> j = {1};
  cellwise_pair_listA->push_back(c, i, j);

  ParticlePairLoopCellwisePairList pl0(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      particle_pair_loop_kernel([](auto NN_A, auto NN_B) {
        NN_A.at(0)++;
        NN_B.at(0)++;
      }),
      Access::A(Access::write(Sym<INT>("NUM_NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NUM_NEIGHBOURS"))));

  pl0.execute();

  auto NN = A->get_cell(Sym<INT>("NUM_NEIGHBOURS"), 0);
  nprint(NN->at(0, 0), NN->at(1, 0));

  sycl_target->free();
}
