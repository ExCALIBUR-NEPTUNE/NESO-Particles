#include "include/test_neso_particles.hpp"

/**
 * Base type for handling the arguments for a pair loop.
 */
template <typename... ARGS>
class ParticlePairLoopArgs : public ParticlePairLoopBase {
protected:
  /// Tuple of the arguments passed to the ParticlePairLoop on construction.
  std::tuple<ARGS...> annotated_args;
  /// Tuple of the arguments with the A/B specification stripped away. These
  /// should be the same types that would be passed to ParticleLoop.
  std::tuple<ARGS...> args;

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

    int cell_start_actual = 0;
    int cell_end_actual = this->cell_count - 1;
    if (cell_start != std::nullopt) {
      cell_start_actual = cell_start.value();
    }
    if (cell_end != std::nullopt) {
      cell_end_actual = cell_end.value();
    }
    const int cell_count_iteration = cell_end_actual - cell_start_actual;

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

      this->event_stack.push(this->sycl_target->queue.parallel_for(
          iteration_set, [=](sycl::item<3> idx) {
            const std::size_t index_cell = idx.get_id(0);
            const std::size_t index_list = idx.get_id(1);
            const std::size_t index_pair = idx.get_id(2);
            const auto *pair_list = &k_pair_lists[index_list];
            const auto num_pairs =
                static_cast<std::size_t>(pair_list->d_pair_counts[index_cell]);
            if (index_pair < num_pairs) {
              const int particle_index_a =
                  pair_list->d_pair_list[index_cell][0][index_pair];
              const int particle_index_b =
                  pair_list->d_pair_list[index_cell][1][index_pair];

              // Now we can create the kernel args.
            }
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

  auto cellwise_pair_listA =
      std::make_shared<DSMC::CellwisePairList>(sycl_target, cell_count);

  // TODO create some pairs here

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

  sycl_target->free();
}
