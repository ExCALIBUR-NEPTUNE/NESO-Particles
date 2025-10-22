#include "include/test_neso_particles.hpp"

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
  CellwisePairListSharedPtr pair_list;
  CellwisePairListAbsolute<ParticleGroup>() = default;
  ~CellwisePairListAbsolute<ParticleGroup>() = default;

  /**
   * Create cell wise pair list where the pair indices correspond to particle
   * layers in two ParticleGroups A and B. A may equal B.
   *
   * @param A First ParticleGroup which suppies the "i" particles.
   * @param B Second ParticleGroup which suppies the "j" particles.
   * @param pair_list Pair list of particle pairs.
   */
  CellwisePairListAbsolute<ParticleGroup>(ParticleGroupSharedPtr A,
                                          ParticleGroupSharedPtr B,
                                          CellwisePairListSharedPtr pair_list)
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

  std::vector<CellwisePairListDevice> h_pair_lists_device;
  std::shared_ptr<BufferDevice<CellwisePairListDevice>> d_pair_lists_device;
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
      : ParticlePairLoopArgs<ARGS...>(nullptr, name, nullptr, nullptr, nullptr,
                                      nullptr, args...),
        pair_lists(pair_lists), kernel(kernel) {

    if (pair_lists.size()) {
      this->num_pair_lists = pair_lists.size();
      this->particle_group_A = get_particle_group(pair_lists[0].A);
      this->particle_group_B = get_particle_group(pair_lists[0].B);
      NESOASSERT(this->particle_group_A->domain ==
                     this->particle_group_B->domain,
                 "Domain missmatch between ParticleGroups.");

      NESOASSERT(this->particle_group_A->sycl_target ==
                     this->particle_group_B->sycl_target,
                 "Compute device missmatch between ParticleGroups.");

      this->sycl_target = this->particle_group_A->sycl_target;
      this->cell_count = this->particle_group_A->domain->mesh->get_cell_count();
      NESOASSERT(this->sycl_target != nullptr, "Bad compute device found");
      this->d_pair_lists_device =
          std::make_shared<BufferDevice<CellwisePairListDevice>>(
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

    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info_A;
    ParticleLoopImplementation::ParticleLoopGlobalInfo global_info_B;

    this->create_global_info(cell_start, cell_end, &global_info_A,
                             &global_info_B);

    const auto cell_start_actual = global_info_A.starting_cell;
    const auto cell_end_actual = global_info_A.bounding_cell;
    const int cell_count_iteration = cell_end_actual - cell_start_actual;

    e0.wait_and_throw();
    if ((max_pair_count > 0) && (this->num_pair_lists > 0) &&
        (cell_count_iteration > 0)) {

      auto k_kernel = this->kernel.kernel;
      auto k_pair_lists = this->d_pair_lists_device->ptr;
      std::size_t local_size = global_info_A.local_size;

      sycl::range<3> local_iteration_set(1, 1, local_size);
      sycl::range<3> global_iteration_set(
          static_cast<std::size_t>(cell_count_iteration),
          static_cast<std::size_t>(this->num_pair_lists),
          static_cast<std::size_t>(
              get_next_multiple(max_pair_count, local_size)));

      sycl::nd_range<3> nd_iteration_set =
          this->sycl_target->device_limits.validate_nd_range(
              sycl::nd_range<3>(global_iteration_set, local_iteration_set));

      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            KernelMasksType kernel_masks;
            this->create_loop_args(cgh, loop_args, kernel_masks, &global_info_A,
                                   &global_info_B);

            cgh.parallel_for(nd_iteration_set, [=](sycl::nd_item<3> idx) {
              const std::size_t index_cell =
                  idx.get_global_id(0) + cell_start_actual;
              const std::size_t index_list = idx.get_global_id(1);
              const std::size_t index_pair = idx.get_global_id(2);

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

TEST(ParticlePairLoop, cellwise_pair_list) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int cell_count = 19;
  auto cellwise_pair_list =
      std::make_shared<CellwisePairList>(sycl_target, cell_count);

  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  std::mt19937 rng(522342);
  std::uniform_int_distribution<int> dist(0, cell_count - 1);

  const int num_samples = 100;

  std::map<int, std::pair<std::vector<int>, std::vector<int>>> h_correct;
  std::vector<int> h_c(num_samples);
  std::vector<int> h_i(num_samples);
  std::vector<int> h_j(num_samples);

  int max_index = -1;
  for (int ix = 0; ix < num_samples; ix++) {
    h_c[ix] = dist(rng);
    h_i[ix] = dist(rng);
    h_j[ix] = dist(rng);
    h_correct[h_c[ix]].first.push_back(h_i[ix]);
    h_correct[h_c[ix]].second.push_back(h_j[ix]);
    max_index = std::max(max_index, h_i[ix]);
    max_index = std::max(max_index, h_j[ix]);
  }

  cellwise_pair_list->push_back(h_c, h_i, h_j);

  {
    auto h_to_test = cellwise_pair_list->host_get();
    auto d_to_test = cellwise_pair_list->get();

    int max_pair_count = 0;

    std::vector<INT> h_pair_counts_es(cell_count);
    sycl_target->queue
        .memcpy(h_pair_counts_es.data(), d_to_test.d_pair_counts_es,
                cell_count * sizeof(INT))
        .wait_and_throw();

    INT es_correct = 0;
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_correct[cx], h_to_test[cx]);
      max_pair_count = std::max(max_pair_count,
                                static_cast<int>(h_to_test[cx].first.size()));
      ASSERT_EQ(es_correct, h_pair_counts_es.at(cx));
      es_correct += d_to_test.h_pair_counts[cx];
    }

    ASSERT_EQ(d_to_test.cell_count, cell_count);
    ASSERT_EQ(d_to_test.max_index, max_index);
    ASSERT_EQ(d_to_test.max_pair_count, max_pair_count);
    ASSERT_EQ(d_to_test.pair_count, num_samples);
  }

  for (int ix = 0; ix < num_samples; ix++) {
    h_c[ix] = dist(rng);
    h_i[ix] = dist(rng);
    h_j[ix] = dist(rng);
    h_correct[h_c[ix]].first.push_back(h_i[ix]);
    h_correct[h_c[ix]].second.push_back(h_j[ix]);
  }

  cellwise_pair_list->push_back(h_c, h_i, h_j);

  {
    auto h_to_test = cellwise_pair_list->host_get();

    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_correct[cx], h_to_test[cx]);
    }
  }

  cellwise_pair_list->clear();
  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  {
    h_c.resize(0);
    h_i.resize(0);
    h_j.resize(0);
    cellwise_pair_list->push_back(h_c, h_i, h_j);
  }

  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  sycl_target->free();
}

TEST(ParticlePairLoop, base) {

  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NEIGHBOURS"), 2);

  particle_loop(
      A,
      [=](auto NN) {
        NN.at(0) = 0;
        NN.at(1) = 1;
      },
      Access::write(Sym<INT>("NEIGHBOURS")))
      ->execute();

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairList>(sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  c.reserve(cell_count * npart_cell / 2);
  i.reserve(cell_count * npart_cell / 2);
  j.reserve(cell_count * npart_cell / 2);

  std::mt19937 rng(9124234 + sycl_target->comm_pair.rank_parent);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    npart_cell = A->get_npart_cell(cellx);
    std::vector<int> pairs(npart_cell);
    std::iota(pairs.begin(), pairs.end(), 0);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    for (int px = 0; px < (npart_cell / 2); px++) {
      c.push_back(cellx);
      i.push_back(pairs.at(2 * px));
      j.push_back(pairs.at(2 * px + 1));
    }
  }

  cellwise_pair_listA->push_back(c, i, j);

  ParticlePairLoopCellwisePairList pl0(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      particle_pair_loop_kernel([](auto ID_i, auto ID_j, auto NN_i, auto NN_j) {
        NN_i.at(0) = ID_j.at(0);
        NN_j.at(0) = ID_i.at(0);
        NN_i.at(1)++;
        NN_j.at(1)++;
      }),
      Access::A(Access::read(Sym<INT>("ID"))),
      Access::B(Access::read(Sym<INT>("ID"))),
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  pl0.execute();

  const std::size_t num_to_test = c.size();
  for (std::size_t pairx = 0; pairx < num_to_test; pairx++) {
    const int cell = c.at(pairx);
    auto ID = A->get_cell(Sym<INT>("ID"), cell);
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cell);

    const int index_i = i.at(pairx);
    const int index_j = j.at(pairx);

    const INT id_i = ID->at(index_i, 0);
    const INT id_j = ID->at(index_j, 0);
    const INT neighbour_i = NN->at(index_i, 0);
    const INT neighbour_j = NN->at(index_j, 0);

    ASSERT_EQ(NN->at(index_i, 1), 2);
    ASSERT_EQ(NN->at(index_j, 1), 2);
    ASSERT_EQ(neighbour_j, id_i);
    ASSERT_EQ(neighbour_i, id_j);
  }

  sycl_target->free();
}

TEST(ParticlePairLoop, foo) {

  int npart_cell = 1000;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<REAL>("FORCE"), 2);

  particle_loop(
      A,
      [=](auto FORCE) {
        FORCE.at(0) = 0;
        FORCE.at(1) = 1;
      },
      Access::write(Sym<REAL>("FORCE")))
      ->execute();

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairList>(sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  c.reserve(cell_count * npart_cell / 2);
  i.reserve(cell_count * npart_cell / 2);
  j.reserve(cell_count * npart_cell / 2);

  std::mt19937 rng(9124234 + sycl_target->comm_pair.rank_parent);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    npart_cell = A->get_npart_cell(cellx);
    std::vector<int> pairs(npart_cell);
    std::iota(pairs.begin(), pairs.end(), 0);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    for (int px = 0; px < (npart_cell / 2); px++) {
      c.push_back(cellx);
      i.push_back(pairs.at(2 * px));
      j.push_back(pairs.at(2 * px + 1));
    }
  }

  cellwise_pair_listA->push_back(c, i, j);

  constexpr REAL sigma = 1.0;
  constexpr REAL sigma2 = sigma * sigma;
  constexpr REAL epsilon = 1.0;
  constexpr REAL CF = -48.0 * epsilon / sigma2;

  ParticlePairLoopCellwisePairList pl0(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      particle_pair_loop_kernel(
          [](auto P_i, auto P_j, auto FORCE_i, auto FORCE_j) {
            const double R0 = P_j.at(0) - P_i.at(0);
            const double R1 = P_j.at(1) - P_i.at(1);

            // distance squared
            const double r2 = R0 * R0 + R1 * R1;

            // (sigma/r)**2, (sigma/r)**4 and  (sigma/r)**6
            const double r_m2 = sigma2 / r2;
            const double r_m4 = r_m2 * r_m2;
            const double r_m6 = r_m4 * r_m2;

            // (sigma/r)**8
            const double r_m8 = r_m4 * r_m4;

            // compute force magnitude
            const double f_tmp = CF * (r_m6 - 0.5) * r_m8;

            // increment force on particle i
            FORCE_i.at(0) = f_tmp * R0;
            FORCE_i.at(1) = f_tmp * R1;
            FORCE_j.at(0) = -f_tmp * R0;
            FORCE_j.at(1) = -f_tmp * R1;
          }),
      Access::A(Access::read(Sym<REAL>("P"))),
      Access::B(Access::read(Sym<REAL>("P"))),
      Access::A(Access::write(Sym<REAL>("FORCE"))),
      Access::B(Access::write(Sym<REAL>("FORCE"))));

  nprint("start");

  for (int stepx = 0; stepx < 1000000; stepx++) {
    nprint(stepx);
    pl0.execute();
  }

  sycl_target->free();
}
