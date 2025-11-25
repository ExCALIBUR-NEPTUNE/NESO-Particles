#include "include/test_neso_particles.hpp"

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

TEST(ParticlePairLoop, iteration_set_size) {

  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  ParticleLoopImplementation::ParticleLoopGlobalInfo global_info;

  global_info.particle_group = A.get();
  global_info.all_cells = true;
  global_info.starting_cell = 0;
  global_info.bounding_cell = cell_count;
  ASSERT_EQ(get_loop_npart(&global_info), A->get_npart_local());
  ASSERT_EQ(get_loop_iteration_set_size(&global_info), A->get_npart_local());
  ASSERT_TRUE(global_info.provided_iteration_set_size);
  ASSERT_EQ(global_info.iteration_set_size, A->get_npart_local());

  global_info.iteration_set_size++;
  ASSERT_EQ(get_loop_iteration_set_size(&global_info),
            A->get_npart_local() + 1);

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

  auto reset_loop = particle_loop(
      A,
      [=](auto NN) {
        NN.at(0) = 0;
        NN.at(1) = 1;
      },
      Access::write(Sym<INT>("NEIGHBOURS")));

  reset_loop->execute();

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

  auto pl0 = particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      [](auto ID_i, auto ID_j, auto NN_i, auto NN_j) {
        NN_i.at(0) = ID_j.at(0);
        NN_j.at(0) = ID_i.at(0);
        NN_i.at(1)++;
        NN_j.at(1)++;
      },
      Access::A(Access::read(Sym<INT>("ID"))),
      Access::B(Access::read(Sym<INT>("ID"))),
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  pl0->execute();

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

  reset_loop->execute();

  const int cell_start = 1;
  const int cell_end = std::max(cell_start, cell_count - 1);
  pl0->execute(cell_start, cell_end);

  for (std::size_t pairx = 0; pairx < num_to_test; pairx++) {
    const int cell = c.at(pairx);
    const int index_i = i.at(pairx);
    const int index_j = j.at(pairx);

    auto ID = A->get_cell(Sym<INT>("ID"), cell);
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cell);

    if ((cell_start <= cell) && (cell < cell_end)) {
      const INT id_i = ID->at(index_i, 0);
      const INT id_j = ID->at(index_j, 0);
      const INT neighbour_i = NN->at(index_i, 0);
      const INT neighbour_j = NN->at(index_j, 0);

      ASSERT_EQ(NN->at(index_i, 1), 2);
      ASSERT_EQ(NN->at(index_j, 1), 2);
      ASSERT_EQ(neighbour_j, id_i);
      ASSERT_EQ(neighbour_i, id_j);
    } else {
      ASSERT_EQ(NN->at(index_i, 0), 0);
      ASSERT_EQ(NN->at(index_j, 0), 0);
      ASSERT_EQ(NN->at(index_i, 1), 1);
      ASSERT_EQ(NN->at(index_j, 1), 1);
    }
  }

  reset_loop->execute();
  cellwise_pair_listA->clear();
  pl0->execute();
  for (std::size_t pairx = 0; pairx < num_to_test; pairx++) {
    const int cell = c.at(pairx);
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cell);

    const int index_i = i.at(pairx);
    const int index_j = j.at(pairx);

    ASSERT_EQ(NN->at(index_i, 1), 1);
    ASSERT_EQ(NN->at(index_j, 1), 1);
  }

  sycl_target->free();
}

TEST(ParticlePairLoop, particle_pair_loop_index) {

  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NEIGHBOURS"), 2);

  auto reset_loop = particle_loop(
      A,
      [=](auto NN) {
        NN.at(0) = 0;
        NN.at(1) = 0;
      },
      Access::write(Sym<INT>("NEIGHBOURS")));

  reset_loop->execute();

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

  auto pl0 = particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      [](auto PAIR_INDEX, auto NN_i, auto NN_j) {
        NN_i.at(0) = PAIR_INDEX.get_loop_linear_index();
        NN_i.at(1) = 1;
        NN_j.at(1) = 2;
      },
      Access::read(ParticlePairLoopIndex{}),
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  pl0->execute();

  std::set<INT> linear_ids_to_test;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cellx);
    const int nrow = NN->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      if (NN->at(rowx, 1) == 1) {
        const INT linear_id = NN->at(rowx, 0);
        ASSERT_FALSE(linear_ids_to_test.count(linear_id));
        linear_ids_to_test.insert(linear_id);
      }
    }
  }

  std::vector<INT> stage_linear_indices(c.size());
  std::iota(stage_linear_indices.begin(), stage_linear_indices.end(), 0);
  std::set<INT> linear_ids_correct;
  for (INT ix : stage_linear_indices) {
    linear_ids_correct.insert(ix);
  }
  ASSERT_EQ(linear_ids_correct, linear_ids_to_test);

  sycl_target->free();
}

TEST(ParticlePairLoop, kernel_rng_base) {

  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NEIGHBOURS"), 2);

  auto reset_loop = particle_loop(
      A,
      [=](auto NN) {
        NN.at(0) = 0;
        NN.at(1) = 0;
      },
      Access::write(Sym<INT>("NEIGHBOURS")));

  reset_loop->execute();

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

  INT rng_index = 1;
  std::set<INT> sampled_values;
  auto rng_lambda = [&]() -> INT {
    INT value = rng_index++;
    sampled_values.insert(value);
    return value;
  };

  const int rng_ncomp = 2;
  auto rng_block_kernel =
      host_per_particle_block_rng<INT>(rng_lambda, rng_ncomp);

  auto pl0 = particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      [](auto PAIR_INDEX, auto RNG, auto NN_i, auto NN_j) {
        bool valid = true;
        NN_i.at(0) = 1;
        NN_i.at(1) = RNG.at(PAIR_INDEX, 0, &valid);
        NN_j.at(0) = 1;
        NN_j.at(1) = RNG.at(PAIR_INDEX, 1, &valid);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(rng_block_kernel),
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  pl0->execute();

  std::set<INT> seen_rng_values;
  const std::size_t num_to_test = c.size();
  for (std::size_t pairx = 0; pairx < num_to_test; pairx++) {
    const int cell = c.at(pairx);
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cell);

    const int index_i = i.at(pairx);
    const int index_j = j.at(pairx);

    const INT NN_i0 = NN->at(index_i, 0);
    const INT NN_j0 = NN->at(index_j, 0);
    const INT NN_i1 = NN->at(index_i, 1);
    const INT NN_j1 = NN->at(index_j, 1);

    ASSERT_EQ(NN_i0, 1);
    ASSERT_EQ(NN_j0, 1);

    ASSERT_FALSE(seen_rng_values.count(NN_i1));
    seen_rng_values.insert(NN_i1);
    ASSERT_FALSE(seen_rng_values.count(NN_j1));
    seen_rng_values.insert(NN_j1);
  }

  ASSERT_TRUE(rng_block_kernel->valid_internal_state());

  auto rng_atomic_kernel =
      host_atomic_block_kernel_rng<INT>(rng_lambda, rng_ncomp);
  auto pl1 = particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup>(A, A, cellwise_pair_listA)},
      [](auto PAIR_INDEX, auto RNG, auto NN_i, auto NN_j) {
        bool valid = true;
        NN_i.at(0) = 1;
        NN_i.at(1) = RNG.at(PAIR_INDEX, 0, &valid);
        NN_j.at(0) = 1;
        NN_j.at(1) = RNG.at(PAIR_INDEX, 1, &valid);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(rng_atomic_kernel),
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  pl1->execute();

  for (std::size_t pairx = 0; pairx < num_to_test; pairx++) {
    const int cell = c.at(pairx);
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cell);

    const int index_i = i.at(pairx);
    const int index_j = j.at(pairx);

    const INT NN_i0 = NN->at(index_i, 0);
    const INT NN_j0 = NN->at(index_j, 0);
    const INT NN_i1 = NN->at(index_i, 1);
    const INT NN_j1 = NN->at(index_j, 1);

    ASSERT_EQ(NN_i0, 1);
    ASSERT_EQ(NN_j0, 1);

    ASSERT_FALSE(seen_rng_values.count(NN_i1));
    seen_rng_values.insert(NN_i1);
    ASSERT_FALSE(seen_rng_values.count(NN_j1));
    seen_rng_values.insert(NN_j1);
  }

  ASSERT_TRUE(rng_atomic_kernel->valid_internal_state());

  sycl_target->free();
}

TEST(CellwisePairListHost, base) {
  const int cell_count = 7;
  CellwisePairListHost cplh(cell_count);

  ASSERT_EQ(cplh.get_next_wave(1, 2), 0);
  cplh.set_next_wave(1, 2, 4);
  ASSERT_EQ(cplh.get_next_wave(1, 2), 4);
  cplh.clear();
  ASSERT_EQ(cplh.get_next_wave(1, 2), 0);

  cplh.push_back(0, 1, 2);
  ASSERT_EQ(cplh.get_next_wave(0, 1), 1);
  ASSERT_EQ(cplh.get_next_wave(0, 2), 1);

  cplh.push_back(0, 4, 3);
  ASSERT_EQ(cplh.get_next_wave(0, 4), 1);
  ASSERT_EQ(cplh.get_next_wave(0, 3), 1);

  cplh.push_back(0, 2, 3);
  ASSERT_EQ(cplh.get_next_wave(0, 0), 0);
  ASSERT_EQ(cplh.get_next_wave(0, 1), 1);
  ASSERT_EQ(cplh.get_next_wave(0, 2), 2);
  ASSERT_EQ(cplh.get_next_wave(0, 3), 2);
  ASSERT_EQ(cplh.get_next_wave(0, 4), 1);
  ASSERT_EQ(cplh.get_next_wave(0, 5), 0);

  auto m = cplh.get();

  ASSERT_EQ(m.count(1), 0);
  ASSERT_EQ(m.at(0).at(0).first, std::vector<int>({1, 3}));
  ASSERT_EQ(m.at(0).at(0).second, std::vector<int>({2, 4}));

  ASSERT_EQ(m.at(0).at(1).first, std::vector<int>({2}));
  ASSERT_EQ(m.at(0).at(1).second, std::vector<int>({3}));
}
