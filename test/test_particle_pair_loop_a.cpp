#include "include/test_neso_particles.hpp"

TEST(ParticlePairLoop, cellwise_pair_list) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int cell_count = 19;
  auto cellwise_pair_list =
      std::make_shared<CellwisePairList>(sycl_target, cell_count);

  {
    auto h_list = cellwise_pair_list->host_get();
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list[cx][0].first.size(), 0);
      ASSERT_EQ(h_list[cx][0].second.size(), 0);
    }
  }

  std::mt19937 rng(522342);
  std::uniform_int_distribution<int> dist(0, cell_count - 1);

  const int num_samples = 100;

  std::map<int, std::pair<std::vector<int>, std::vector<int>>> h_correct;
  std::vector<int> h_c(num_samples);
  std::vector<int> h_i(num_samples);
  std::vector<int> h_j(num_samples);

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
    auto d_to_test = cellwise_pair_list->get();

    int max_pair_count = 0;

    std::vector<INT> h_pair_counts_es(cell_count);
    sycl_target->queue
        .memcpy(h_pair_counts_es.data(), d_to_test.d_pair_counts_es,
                cell_count * sizeof(INT))
        .wait_and_throw();

    INT es_correct = 0;
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_correct[cx], h_to_test[cx][0]);
      max_pair_count = std::max(
          max_pair_count, static_cast<int>(h_to_test[cx][0].first.size()));
      ASSERT_EQ(es_correct, h_pair_counts_es.at(cx));
      es_correct += d_to_test.h_pair_counts[cx];
    }

    ASSERT_EQ(d_to_test.cell_count, cell_count);
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
      ASSERT_EQ(h_correct[cx], h_to_test[cx][0]);
    }
  }

  cellwise_pair_list->clear();
  {
    auto h_list = cellwise_pair_list->host_get();
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list[cx][0].first.size(), 0);
      ASSERT_EQ(h_list[cx][0].second.size(), 0);
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
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list[cx][0].first.size(), 0);
      ASSERT_EQ(h_list[cx][0].second.size(), 0);
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
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
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
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
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
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
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

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  auto rng_atomic_kernel =
      host_atomic_block_kernel_rng<INT>(rng_lambda, rng_ncomp);
  auto pl1 = particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [=](auto PAIR_INDEX, auto RNG, auto NN_i, auto NN_j) {
        bool valid = true;
        NN_i.at(0) = 1;
        NN_i.at(1) = RNG.at(PAIR_INDEX, 0, &valid);
        NESO_KERNEL_ASSERT(valid, k_ep);
        NN_j.at(0) = 1;
        NN_j.at(1) = RNG.at(PAIR_INDEX, 1, &valid);
        NESO_KERNEL_ASSERT(valid, k_ep);
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(rng_atomic_kernel),
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  pl1->execute();

  ASSERT_FALSE(ep.get_flag());

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

TEST(CellwisePairListHost, host) {
  const int cell_count = 7;
  CellwisePairListHost cplh(cell_count);

  cplh.push_back(1, 6, 7);
  ASSERT_EQ(cplh.get_next_wave(1, 2), 0);
  cplh.set_next_wave(1, 2, 1);
  ASSERT_EQ(cplh.get_next_wave(1, 2), 1);
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

TEST(CellwisePairListHost, device) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int cell_count = 7;
  auto cplh = std::make_shared<CellwisePairListHost>(cell_count);
  auto cpl = std::make_shared<CellwisePairList>(sycl_target, cell_count);

  {
    cpl->clear();
    cpl->set(cplh);
    auto m = cpl->get();
    ASSERT_EQ(m.cell_count, cell_count);
    ASSERT_EQ(m.max_pair_count, 0);
    ASSERT_EQ(m.pair_count, 0);

    std::vector<int> h_wave_count(cell_count);
    sycl_target->queue
        .memcpy(h_wave_count.data(), m.d_wave_count, cell_count * sizeof(int))
        .wait_and_throw();

    for (int cellx = 0; cellx < cell_count; cellx++) {
      ASSERT_EQ(m.h_wave_count[cellx], 0);
      ASSERT_EQ(m.h_pair_counts[cellx], 0);
      ASSERT_EQ(h_wave_count[cellx], 0);
    }
  }

  {
    cpl->clear();

    const int num_samples = 37;
    std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
    std::uniform_int_distribution<int> dist_cell(0, cell_count - 1);

    std::vector<int> h_npart_cell(cell_count);
    std::vector<std::uniform_int_distribution<int>> dist_indices(cell_count);
    for (int cellx = 0; cellx < cell_count; cellx++) {
      h_npart_cell.at(cellx) = cellx + 2;
      dist_indices.at(cellx) =
          std::uniform_int_distribution<int>(0, h_npart_cell.at(cellx) - 1);
    }

    std::set<std::tuple<int, int, int>> pairs_correct;
    std::map<std::tuple<int, int, int>, int> pairs_count_correct;

    for (int samplex = 0; samplex < num_samples; samplex++) {

      bool valid = false;
      const int cell = dist_cell(rng);
      const int i = dist_indices.at(cell)(rng);
      int j = -1;
      do {
        j = dist_indices.at(cell)(rng);
        valid = i != j;
      } while (!valid);
      cplh->push_back(cell, i, j);
      const int oi = i < j ? i : j;
      const int oj = i < j ? j : i;
      pairs_correct.insert({cell, oi, oj});
      pairs_count_correct[{cell, oi, oj}]++;
    }

    auto h = cplh->get();

    std::map<std::tuple<int, int, int>, int> pairs_count_to_test;

    for (auto &cell_map : h) {
      const int cell = cell_map.first;
      for (auto &wave_map : cell_map.second) {
        std::set<int> seen_indices;
        const int num_pairs = wave_map.second.first.size();
        for (int px = 0; px < num_pairs; px++) {
          const int i = wave_map.second.first.at(px);
          const int j = wave_map.second.second.at(px);
          const int oi = i < j ? i : j;
          const int oj = i < j ? j : i;
          ASSERT_TRUE(pairs_correct.count({cell, oi, oj}));
          pairs_count_to_test[{cell, oi, oj}]++;
          ASSERT_EQ(seen_indices.count(oi), 0);
          ASSERT_EQ(seen_indices.count(oj), 0);
          seen_indices.insert(oi);
          seen_indices.insert(oj);
        }
      }
    }
    ASSERT_EQ(pairs_count_correct, pairs_count_to_test);

    cpl->set(cplh);
    h = cpl->host_get();

    pairs_count_to_test.clear();
    for (auto &cell_map : h) {
      const int cell = cell_map.first;
      for (auto &wave_map : cell_map.second) {
        const int num_pairs = wave_map.second.first.size();
        for (int px = 0; px < num_pairs; px++) {
          const int i = wave_map.second.first.at(px);
          const int j = wave_map.second.second.at(px);
          const int oi = i < j ? i : j;
          const int oj = i < j ? j : i;
          ASSERT_TRUE(pairs_correct.count({cell, oi, oj}));
          pairs_count_to_test[{cell, oi, oj}]++;
        }
      }
    }
    ASSERT_EQ(pairs_count_correct, pairs_count_to_test);

    {
      auto d = cpl->get();
      std::vector<int> h_wave_count(cell_count);
      sycl_target->queue
          .memcpy(h_wave_count.data(), d.d_wave_count, cell_count * sizeof(int))
          .wait_and_throw();

      int max_wave_count = 0;
      for (int cellx = 0; cellx < cell_count; cellx++) {
        ASSERT_EQ(h_wave_count[cellx], d.h_wave_count[cellx]);
        max_wave_count = std::max(max_wave_count, h_wave_count[cellx]);
      }

      const int N = max_wave_count * cell_count;
      std::vector<int> h_wave_offsets(N);
      std::vector<int> h_pair_counts(N);
      std::vector<INT> h_pair_counts_es(N);

      sycl_target->queue
          .memcpy(h_wave_offsets.data(), d.d_wave_offsets, N * sizeof(int))
          .wait_and_throw();
      sycl_target->queue
          .memcpy(h_pair_counts.data(), d.d_pair_counts, N * sizeof(int))
          .wait_and_throw();
      sycl_target->queue
          .memcpy(h_pair_counts_es.data(), d.d_pair_counts_es, N * sizeof(INT))
          .wait_and_throw();

      int pair_count = 0;

      std::set<int> linear_to_test;
      std::set<int> linear_correct;
      for (int ix = 0; ix < num_samples; ix++) {
        linear_correct.insert(ix);
      }

      for (int cellx = 0; cellx < cell_count; cellx++) {
        const int wave_count = h_wave_count[cellx];
        int wave_offset = 0;
        for (int wavex = 0; wavex < wave_count; wavex++) {
          const int num_pairs = h_pair_counts[wavex * cell_count + cellx];
          ASSERT_EQ(num_pairs, d.h_pair_counts[wavex * cell_count + cellx]);
          pair_count += num_pairs;
          ASSERT_EQ(h_wave_offsets[wavex * cell_count + cellx], wave_offset);
          wave_offset += num_pairs;

          for (int ix = 0; ix < num_pairs; ix++) {
            const int linear_index =
                h_pair_counts_es[wavex * cell_count + cellx] + ix;
            linear_to_test.insert(linear_index);
          }
        }
      }
      ASSERT_EQ(pair_count, num_samples);
      ASSERT_EQ(linear_to_test, linear_correct);
    }
  }

  sycl_target->free();
}

TEST(ParticlePairLoop, cell_wise_pair_list_waves) {

  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  A->add_particle_dat(Sym<INT>("NEIGHBOURS"), 1);

  auto reset_loop = particle_loop(
      A, [=](auto NN) { NN.at(0) = 0; }, Access::write(Sym<INT>("NEIGHBOURS")));

  reset_loop->execute();

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairList>(sycl_target, cell_count);

  auto pl0 = particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [](auto NN_i, auto NN_j) {
        NN_i.at(0)++;
        NN_j.at(0)++;
      },
      Access::A(Access::write(Sym<INT>("NEIGHBOURS"))),
      Access::B(Access::write(Sym<INT>("NEIGHBOURS"))));

  pl0->execute();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cellx);
    const int nrow = NN->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      ASSERT_EQ(NN->at(rowx, 0), 0);
    }
  }

  auto cplh = std::make_shared<CellwisePairListHost>(cell_count);

  const int num_samples = 37000;
  std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
  std::uniform_int_distribution<int> dist_cell(0, cell_count - 1);
  std::uniform_real_distribution<REAL> dist_row(0.0, 1.0);

  std::vector<int> cells;
  std::vector<int> pairs_i;
  std::vector<int> pairs_j;

  cells.reserve(num_samples);
  pairs_i.reserve(num_samples);
  pairs_j.reserve(num_samples);

  std::map<std::tuple<int, int>, int> num_neighbours_correct;

  for (int samplex = 0; samplex < num_samples; samplex++) {
    bool valid = false;
    const int cell = dist_cell(rng);
    const int npart_cell = A->get_npart_cell(cell);

    auto lambda_sample_index = [&]() -> int {
      const REAL u = dist_row(rng);
      const int row0 = u * npart_cell;
      const int row1 = std::min(row0, npart_cell - 1);
      return row1;
    };

    const int i = lambda_sample_index();
    int j = -1;
    do {
      j = lambda_sample_index();
      valid = i != j;
    } while (!valid);
    cplh->push_back(cell, i, j);
    const int oi = i < j ? i : j;
    const int oj = i < j ? j : i;
    cells.push_back(cell);
    pairs_i.push_back(oi);
    pairs_j.push_back(oj);

    std::tuple<int, int> key_i = {cell, i};
    std::tuple<int, int> key_j = {cell, j};

    for (auto k : {key_i, key_j}) {
      if (num_neighbours_correct.count(k)) {
        num_neighbours_correct[k]++;
      } else {
        num_neighbours_correct[k] = 1;
      }
    }
  }

  cellwise_pair_listA->set(cplh);
  pl0->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto NN = A->get_cell(Sym<INT>("NEIGHBOURS"), cellx);
    const int nrow = NN->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      std::tuple<int, int> k = {cellx, rowx};
      if (num_neighbours_correct.count(k)) {
        ASSERT_EQ(NN->at(rowx, 0), num_neighbours_correct[k]);
      } else {
        ASSERT_EQ(NN->at(rowx, 0), 0);
      }
    }
  }

  std::vector<int> h_masks(num_samples);
  std::fill(h_masks.begin(), h_masks.end(), 1);
  BufferDevice d_masks(sycl_target, h_masks);
  auto *k_masks = d_masks.ptr;

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [=](auto INDEX) { k_masks[INDEX.get_loop_linear_index()] += 1; },
      Access::read(ParticlePairLoopIndex{}))
      ->execute();

  h_masks = d_masks.get();
  for (int ix = 0; ix < num_samples; ix++) {
    ASSERT_EQ(h_masks.at(ix), 2);
  }

  sycl_target->free();
  A->domain->mesh->free();
}
