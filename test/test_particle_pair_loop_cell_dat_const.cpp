#include "include/test_neso_particles.hpp"

TEST(ParticlePairLoop, cell_dat_const) {

  int npart_cell = 10;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A, sycl_target, cell_count] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);

  A->add_particle_dat(Sym<INT>("INDEX"), 1);
  particle_loop(
      A, [=](auto INDEX) { INDEX.at(0) = -1; },
      Access::write(Sym<INT>("INDEX")))
      ->execute();

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairListSimple>(sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  c.reserve(cell_count * npart_cell / 2);
  i.reserve(cell_count * npart_cell / 2);
  j.reserve(cell_count * npart_cell / 2);

  std::mt19937 rng(9124234 + sycl_target->comm_pair.rank_parent);

  std::vector<int> h_pair_counts(cell_count);

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
    h_pair_counts.at(cellx) = (npart_cell / 2);
  }

  cellwise_pair_listA->push_back(c, i, j);

  auto cdc0 =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 1, 2);

  auto h_cdc0 = cdc0->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int dx = 0; dx < 2; dx++) {
      h_cdc0.at(cellx)->at(0, dx) = cellx + dx * 0.1;
    }
  }
  cdc0->set_all_cells(h_cdc0);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_pair_loop(
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [=](auto PAIR_INDEX, auto CDC) {
        for (int dx = 0; dx < 2; dx++) {
          NESO_KERNEL_ASSERT(CDC.at(0, dx) == PAIR_INDEX.cell + dx * 0.1, k_ep);
        }
      },
      Access::read(ParticlePairLoopIndex{}), Access::read(cdc0))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  auto cdc1 =
      std::make_shared<CellDatConst<int>>(sycl_target, cell_count, 1, 1);
  cdc1->fill(0);

  particle_pair_loop(
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [=](auto INDEX, auto CDC) { INDEX.at(0) = CDC.fetch_add(0, 0, 1); },
      Access::A(Access::write(Sym<INT>("INDEX"))), Access::add(cdc1))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  auto h_cdc1 = cdc1->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(h_pair_counts.at(cellx), h_cdc1.at(cellx)->at(0, 0));
  }

  particle_pair_loop(
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [=](auto PI, auto INDEX, auto CDC) {
        if (INDEX.at(0) == 0) {
          CDC.at(0, 0) = PI.layer;
        }
      },
      Access::A(Access::read(ParticlePairLoopIndex{})),
      Access::A(Access::read(Sym<INT>("INDEX"))), Access::write(cdc1))
      ->execute();

  particle_loop(
      A,
      [=](auto PI, auto INDEX, auto CDC) {
        if (INDEX.at(0) == 0) {
          NESO_KERNEL_ASSERT(CDC.at(0, 0) == PI.layer, k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("INDEX")),
      Access::read(cdc1))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  BufferDevice<int> d_buffer(sycl_target, cell_count);
  int *k_buffer = d_buffer.ptr;

  cdc1->fill(100000);
  sycl_target->queue.fill(k_buffer, 100000, cell_count).wait_and_throw();

  particle_pair_loop(
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [=](auto PI, auto CDC) {
        CDC.fetch_min(0, 0, PI.layer);
        atomic_fetch_min(k_buffer + PI.cell, static_cast<int>(PI.layer));
      },
      Access::A(Access::read(ParticlePairLoopIndex{})), Access::min(cdc1))
      ->execute();

  auto h_buffer = d_buffer.get();

  h_cdc1 = cdc1->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(h_cdc1.at(cellx)->at(0, 0), h_buffer.at(cellx));
  }

  cdc1->fill(-100000);
  sycl_target->queue.fill(k_buffer, -100000, cell_count).wait_and_throw();

  particle_pair_loop(
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_listA)},
      [=](auto PI, auto CDC) {
        CDC.fetch_max(0, 0, PI.layer);
        atomic_fetch_max(k_buffer + PI.cell, static_cast<int>(PI.layer));
      },
      Access::A(Access::read(ParticlePairLoopIndex{})), Access::max(cdc1))
      ->execute();

  h_buffer = d_buffer.get();

  h_cdc1 = cdc1->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(h_cdc1.at(cellx)->at(0, 0), h_buffer.at(cellx));
  }

  sycl_target->free();
  A->domain->mesh->free();
}
