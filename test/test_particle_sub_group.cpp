#include "include/test_neso_particles.hpp"

namespace {

const int ndim = 2;

auto particle_loop_common(const int N = 1093) {
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;

  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1),
                             ParticleProp(Sym<INT>("MARKER"), 1)};

  auto A = make_test_obj<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int rank = sycl_target->comm_pair.rank_parent;
  const INT id_offset = rank * N;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
    initial_distribution[Sym<INT>("ID")][px][0] = px + id_offset;
  }

  A->add_particles_local(initial_distribution);
  parallel_advection_initialisation(A, 16);

  auto ccb = std::make_shared<CartesianCellBin>(
      sycl_target, mesh, A->position_dat, A->cell_id_dat);

  ccb->execute();
  A->cell_move();

  return A;
}

class TestParticleSubGroup : public ParticleSubGroup {
public:
  template <typename KERNEL, typename... ARGS>
  TestParticleSubGroup(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                       ARGS... args)
      : ParticleSubGroup(particle_group, kernel, args...) {}

  template <typename KERNEL, typename... ARGS>
  TestParticleSubGroup(ParticleSubGroupSharedPtr particle_group, KERNEL kernel,
                       ARGS... args)
      : ParticleSubGroup(particle_group, kernel, args...) {}

  inline int test_get_cells_layers(std::vector<INT> &cells,
                                   std::vector<INT> &layers) {
    return get_cells_layers(cells, layers);
  }
};

} // namespace

TEST(ParticleSubGroup, version_tracker) {

  ParticleDatVersionT v0;
  ParticleDatVersionT v1;
  EXPECT_FALSE(v0 < v1);
  ParticleDatVersionT r0(Sym<REAL>("A"));
  ParticleDatVersionT r1(Sym<REAL>("A"));
  ParticleDatVersionT r2(Sym<REAL>("B"));
  ParticleDatVersionT i0(Sym<INT>("A"));
  ParticleDatVersionT i1(Sym<INT>("A"));
  ParticleDatVersionT i2(Sym<INT>("B"));
  EXPECT_TRUE(v0 < r0);
  EXPECT_TRUE(v0 < i0);
  EXPECT_FALSE(r0 < i0);
  EXPECT_FALSE(r0 < r1);
  EXPECT_TRUE(r0 < r2);
  EXPECT_FALSE(i0 < i1);
  EXPECT_TRUE(i0 < i2);
  EXPECT_FALSE(r0 < i0);
  EXPECT_TRUE(i0 < r0);

  ParticleDatVersionT v2;
  v2 = Sym<INT>("C");
  ASSERT_TRUE(v2.index == 0);
  ASSERT_TRUE(v2.si.name == "C");
  ParticleDatVersionT v3;
  v3 = Sym<REAL>("C");
  ASSERT_TRUE(v3.index == 1);
  ASSERT_TRUE(v3.sr.name == "C");
  v2 = v3;
  ASSERT_TRUE(v2.index == 1);
  ASSERT_TRUE(v2.sr.name == "C");
  v3 = Sym<INT>("D");
  ASSERT_TRUE(v3.index == 0);
  ASSERT_TRUE(v3.si.name == "D");
}

TEST(ParticleSubGroup, selector) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  TestParticleSubGroup aa(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  aa.create();

  std::vector<INT> cells;
  std::vector<INT> layers;

  const int num_particles = aa.test_get_cells_layers(cells, layers);
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);

    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    EXPECT_TRUE((*id)[0][layerx] % 2 == 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

namespace {
struct TestSubGroupSelector
    : public ParticleSubGroupImplementation::SubGroupSelector {

  template <typename... T>
  TestSubGroupSelector(T... args)
      : ParticleSubGroupImplementation::SubGroupSelector(args...) {}

  inline std::shared_ptr<CellDat<INT>> get_map() {
    return this->map_cell_to_particles;
  }
};

struct TestCellSubGroupSelector
    : public ParticleSubGroupImplementation::CellSubGroupSelector {

  template <typename... T>
  TestCellSubGroupSelector(T... args)
      : ParticleSubGroupImplementation::CellSubGroupSelector(args...) {}

  inline std::shared_ptr<CellDat<INT>> get_map() {
    return this->map_cell_to_particles;
  }
};

template <typename T>
inline bool check_selector(ParticleGroupSharedPtr particle_group,
                           std::shared_ptr<T> selector, std::vector<int> &cells,
                           std::vector<int> &layers) {
  bool status = true;
  auto s = selector->get();

  auto lambda_check_eq = [&](auto a, auto b) { status = status && (a == b); };
  auto lambda_check_true = [&](const bool a) { status = status && a; };

  lambda_check_eq(static_cast<std::size_t>(cells.size()), layers.size());
  lambda_check_eq(static_cast<std::size_t>(s.npart_local), layers.size());
  lambda_check_eq(s.ncell, particle_group->domain->mesh->get_cell_count());

  std::map<int, std::set<int>> map_cells_layers;
  for (int ix = 0; ix < s.npart_local; ix++) {
    auto c = cells.at(ix);
    auto l = layers.at(ix);
    lambda_check_true(!static_cast<bool>(map_cells_layers[c].count(l)));
    map_cells_layers[c].insert(l);
  }

  auto sycl_target = particle_group->sycl_target;
  std::vector<int> tmp_int(s.ncell);
  sycl_target->queue
      .memcpy(tmp_int.data(), s.d_npart_cell, s.ncell * sizeof(int))
      .wait_and_throw();

  for (int cx = 0; cx < s.ncell; cx++) {
    lambda_check_eq(static_cast<std::size_t>(s.h_npart_cell[cx]),
                    map_cells_layers[cx].size());
    lambda_check_eq(static_cast<std::size_t>(tmp_int.at(cx)),
                    map_cells_layers[cx].size());
  }

  std::vector<INT> tmp_INT(s.ncell);
  sycl_target->queue
      .memcpy(tmp_INT.data(), s.d_npart_cell_es, s.ncell * sizeof(INT))
      .wait_and_throw();

  INT total = 0;
  for (int cx = 0; cx < s.ncell; cx++) {
    lambda_check_eq(tmp_INT.at(cx), total);
    total += s.h_npart_cell[cx];
  }

  auto map_device = selector->get_map();

  for (int cx = 0; cx < s.ncell; cx++) {
    const int nrow = s.h_npart_cell[cx];
    lambda_check_true(map_device->nrow.at(cx) >= nrow);
    std::set<int> in_map;
    for (int rx = 0; rx < nrow; rx++) {
      in_map.insert(map_device->get_value(cx, rx, 0));
    }
    lambda_check_eq(in_map, map_cells_layers.at(cx));
  }

  return status;
}

} // namespace

TEST(ParticleSubGroup, selector_get_even_id) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto s = std::make_shared<TestSubGroupSelector>(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  int cell_count = A->domain->mesh->get_cell_count();
  for (int cx = 0; cx < cell_count; cx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cx);
    for (int rx = 0; rx < ID->nrow; rx++) {
      if (ID->at(rx, 0) % 2 == 0) {
        cells.push_back(cx);
        layers.push_back(rx);
      }
    }
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, selector_get_even_id_even_cell) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto aa = particle_sub_group(
      A, [=](auto CELL_ID) { return ((CELL_ID.at(0) % 2) == 0); },
      Access::read(Sym<INT>("CELL_ID")));

  auto s = std::make_shared<TestSubGroupSelector>(
      aa, [=](auto ID) { return ((ID.at(0) % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  int cell_count = A->domain->mesh->get_cell_count();
  for (int cx = 0; cx < cell_count; cx++) {
    auto ID = A->get_cell(Sym<INT>("ID"), cx);
    auto CELL_ID = A->get_cell(Sym<INT>("CELL_ID"), cx);
    for (int rx = 0; rx < ID->nrow; rx++) {
      if ((ID->at(rx, 0) % 2 == 0) && (CELL_ID->at(rx, 0) % 2 == 0)) {
        cells.push_back(cx);
        layers.push_back(rx);
      }
    }
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, selector_get_cell) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  int cell_count = A->domain->mesh->get_cell_count();
  int cx = cell_count - 1;

  auto s = std::make_shared<TestCellSubGroupSelector>(A, cx);

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  const int nrow = A->get_npart_cell(cx);
  for (int rx = 0; rx < nrow; rx++) {
    cells.push_back(cx);
    layers.push_back(rx);
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, selector_get_cell_even_id) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  int cell_count = A->domain->mesh->get_cell_count();
  int cx = cell_count - 1;

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ((ID.at(0) % 2) == 0); },
      Access::read(Sym<INT>("ID")));
  auto s = std::make_shared<TestCellSubGroupSelector>(aa, cx);

  std::vector<int> cells;
  std::vector<int> layers;
  cells.reserve(A->get_npart_local());
  layers.reserve(A->get_npart_local());

  const int nrow = A->get_npart_cell(cx);
  auto ID = A->get_cell(Sym<INT>("ID"), cx);
  for (int rx = 0; rx < nrow; rx++) {
    if (ID->at(rx, 0) % 2 == 0) {
      cells.push_back(cx);
      layers.push_back(rx);
    }
  }

  ASSERT_TRUE(check_selector(A, s, cells, layers));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, particle_loop) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] == 2; }, Access::read(Sym<INT>("ID")));

  GlobalArray<int> counter(sycl_target, 1, 0);

  auto pl_counter = particle_loop(
      aa, [=](Access::GlobalArray::Add<int> G1) { G1.add(0, 1); },
      Access::add(counter));

  pl_counter->execute();

  auto vector_c = counter.get();
  EXPECT_EQ(vector_c.at(0), 1);

  auto bb = std::make_shared<TestParticleSubGroup>(
      A, [=](auto ID) { return ((ID[0] % 2) == 0); },
      Access::read(Sym<INT>("ID")));

  LocalArray<int> counter2(sycl_target, 2, 0);

  auto pl_counter2 = particle_loop(
      std::dynamic_pointer_cast<ParticleSubGroup>(bb),
      [=](auto G1, auto ID, auto MARKER) {
        G1.fetch_add(0, 1);
        G1.fetch_add(1, ID[0]);
        MARKER[0] = 1;
      },
      Access::add(counter2), Access::read(Sym<INT>("ID")),
      Access::write(Sym<INT>("MARKER")));

  pl_counter2->execute();
  auto vector_b = counter2.get();

  std::vector<INT> cells;
  std::vector<INT> layers;
  const int num_particles = bb->test_get_cells_layers(cells, layers);

  EXPECT_EQ(num_particles, vector_b.at(0));

  int id_counter = 0;
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);

    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    id_counter += (*id)[0][layerx];
  }

  EXPECT_EQ(id_counter, vector_b.at(1));

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    const int nrow = id->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      if ((*id)[0][rowx] % 2 == 0) {
        EXPECT_EQ((*marker)[0][rowx], 1);
      } else {
        EXPECT_EQ((*marker)[0][rowx], 0);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, creating) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return (ID[0] % 2) == 0; },
      Access::read(Sym<INT>("ID")));

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->cell_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->local_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->global_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->hybrid_move();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  auto remover = std::make_shared<ParticleRemover>(A->sycl_target);
  const int npart0 = A->get_npart_local();
  remover->remove(A, A->get_dat(Sym<INT>("ID")), 1);
  const int npart1 = A->get_npart_local();

  if (npart0 != npart1) {
    EXPECT_TRUE(aa->create_if_required());
  }
  EXPECT_FALSE(aa->create_if_required());

  auto la = std::make_shared<LocalArray<INT>>(sycl_target, 1);
  particle_loop(
      aa, [=](auto LA) { LA.fetch_add(0, 1); }, Access::add(la))
      ->execute();
  auto lav = la->get();
  const int npart_local = lav.at(0);
  int npart_min;
  MPICHK(MPI_Allreduce(&npart_local, &npart_min, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD));
  if (npart_min == 0) {
    return;
  }

  auto p0 = particle_loop(
      A, [](auto /*ID*/, auto V) { V[0] += 0.0001; },
      Access::read(Sym<INT>("ID")), Access::write(Sym<REAL>("V")));
  p0->execute();
  EXPECT_FALSE(aa->create_if_required());

  auto p1 = particle_loop(
      aa, [](auto /*ID*/, auto V) { V[0] += 0.0001; },
      Access::read(Sym<INT>("ID")), Access::write(Sym<REAL>("V")));
  p1->execute();
  EXPECT_FALSE(aa->create_if_required());

  p1->execute();
  p1->execute();
  EXPECT_FALSE(aa->create_if_required());

  auto p2 = particle_loop(
      A, [](auto ID) { ID[0] += 2; }, Access::write(Sym<INT>("ID")));
  p2->execute();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  std::vector<Sym<INT>> sym_vector_id = {Sym<INT>("ID")};
  aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID.at(0, 0) % 2 == 0; },
      Access::read(sym_vector(A, sym_vector_id)));

  auto p3 = particle_loop(
      aa, [](auto ID) { ID[0] += 2; }, Access::write(Sym<INT>("ID")));
  p3->execute();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  ParticleSet distribution(1, A->particle_spec);
  A->add_particles_local(distribution);

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  // This sets the P pointer to be "possibly cached"
  A->get_dat(Sym<REAL>("P"))->cell_dat.device_ptr();
  EXPECT_FALSE(aa->create_if_required());

  // This sets the ID pointer to be "possibly cached"
  A->get_dat(Sym<INT>("ID"))->cell_dat.device_ptr();
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_TRUE(aa->create_if_required());

  EXPECT_EQ(aa->static_status(), false);
  EXPECT_EQ(aa->static_status(false), false);
  EXPECT_EQ(aa->static_status(true), true);
  EXPECT_EQ(aa->static_status(), true);
  EXPECT_TRUE(aa->is_valid());

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, static_valid) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto lambda_make_aa = [&]() {
    return static_particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));
  };

  auto aa = lambda_make_aa();
  EXPECT_TRUE(aa->is_valid());

  particle_loop(
      aa, [=](auto ID) { ID.at(0) += 2; }, Access::write(Sym<INT>("ID")))
      ->execute();
  EXPECT_TRUE(aa->is_valid());

  aa = lambda_make_aa();
  A->cell_move();
  EXPECT_TRUE(!aa->is_valid());

  aa = lambda_make_aa();
  A->hybrid_move();
  EXPECT_TRUE(!aa->is_valid());

  aa = lambda_make_aa();
  A->add_particles_local(aa);
  EXPECT_TRUE(!aa->is_valid());

  auto bb = static_particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 4 == 0; },
      Access::read(Sym<INT>("ID")));
  aa = lambda_make_aa();
  A->remove_particles(bb);
  EXPECT_TRUE(!aa->is_valid());

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, particle_loop_index) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  auto pl_reset = particle_loop(
      A, [=](auto MARKER) { MARKER[0] = -1; },
      Access::write(Sym<INT>("MARKER")));
  pl_reset->execute();

  auto pl = particle_loop(
      aa,
      [=](auto MARKER, auto index) {
        MARKER[0] = index.get_local_linear_index();
      },
      Access::write(Sym<INT>("MARKER")), Access::read(ParticleLoopIndex{}));
  pl->execute();

  INT index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      const INT ix = (*id)[0][rowx];
      if (ix % 2 == 0) {
        ASSERT_EQ(mx, index);
      } else {
        ASSERT_EQ(mx, -1);
      }
      index++;
    }
  }

  std::vector<INT> gav = {0};
  auto ga = std::make_shared<LocalArray<INT>>(sycl_target, gav);
  pl = particle_loop(
      aa,
      [=](auto MARKER, auto index, auto GA) {
        MARKER[0] = index.get_loop_linear_index();
        GA.fetch_add(0, 1);
      },
      Access::write(Sym<INT>("MARKER")), Access::read(ParticleLoopIndex{}),
      Access::add(ga));

  pl_reset->execute();
  pl->execute();
  gav = ga->get();
  const int npart_la = gav.at(0);

  std::set<INT> found_indices;
  const INT npart = aa->get_npart_local();
  index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      const INT ix = (*id)[0][rowx];
      if (ix % 2 == 0) {
        ASSERT_TRUE(mx < npart);
        ASSERT_TRUE(mx > -1);
        found_indices.insert(mx);
        index++;
      } else {
        ASSERT_EQ(mx, -1);
      }
    }
  }
  ASSERT_EQ(npart_la, npart);
  ASSERT_EQ(index, npart);
  ASSERT_EQ(found_indices.size(), npart);

  auto loop_indexing = particle_loop(
      aa,
      [](auto index, auto MARKER) {
        MARKER.at(0) = index.get_loop_linear_index();
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("MARKER")));

  for (int cx = 0; cx < cell_count; cx++) {
    loop_indexing->execute(cx);
  }

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    index = 0;
    found_indices.clear();

    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      const INT ix = (*id)[0][rowx];
      if (ix % 2 == 0) {
        ASSERT_TRUE(mx < nrow);
        ASSERT_TRUE(mx > -1);
        found_indices.insert(mx);
        index++;
      } else {
        ASSERT_EQ(mx, -1);
      }
    }
    ASSERT_EQ(found_indices.size(), index);
    ASSERT_EQ(found_indices.size(), aa->get_npart_cell(cellx));
  }

  auto loop_sub_indexing = particle_loop(
      aa,
      [](auto index, auto MARKER) {
        MARKER.at(0) = index.get_sub_linear_index();
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("MARKER")));
  for (int cx = 0; cx < cell_count; cx++) {
    loop_sub_indexing->execute(cx);
  }

  std::set<INT> sub_indices, sub_correct;
  const int npart_local = aa->get_npart_local();
  for (int ix = 0; ix < npart_local; ix++) {
    sub_correct.insert(ix);
  }

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_cell(Sym<INT>("MARKER"), cellx);
    auto id = A->get_cell(Sym<INT>("ID"), cellx);
    const int nrow = marker->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      if (id->at(rowx, 0) % 2 == 0) {
        const INT mx = marker->at(rowx, 0);
        sub_indices.insert(mx);
      }
    }
  }
  ASSERT_EQ(sub_correct, sub_indices);

  loop_sub_indexing->execute();

  sub_indices.clear();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_cell(Sym<INT>("MARKER"), cellx);
    auto id = A->get_cell(Sym<INT>("ID"), cellx);
    const int nrow = marker->nrow;
    for (int rowx = 0; rowx < nrow; rowx++) {
      if (id->at(rowx, 0) % 2 == 0) {
        const INT mx = marker->at(rowx, 0);
        sub_indices.insert(mx);
      }
    }
  }
  ASSERT_EQ(sub_correct, sub_indices);

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, whole_group) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto pl_set = particle_loop(
      A, [](auto m) { m.at(0) = 2; }, Access::write(Sym<INT>("MARKER")));
  pl_set->execute();

  auto aa = std::make_shared<ParticleSubGroup>(A);
  ASSERT_TRUE(aa->is_entire_particle_group());

  auto pl_set_aa = particle_loop(
      aa, [](auto m) { m.at(0) -= 1; }, Access::write(Sym<INT>("MARKER")));
  pl_set_aa->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto marker = A->get_dat(Sym<INT>("MARKER"))->cell_dat.get_cell(cellx);
    const int nrow = marker->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      ASSERT_EQ(mx, 1);
    }
  }

  ASSERT_EQ(A->get_npart_local(), aa->get_npart_local());

  A->remove_particles(aa);
  ASSERT_EQ(A->get_npart_local(), 0);
  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(A->get_npart_cell(cx), 0);
    ASSERT_EQ(aa->get_npart_cell(cx), 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, remove_particles) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto bb = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  ASSERT_EQ(bb->get_npart_local() + aa->get_npart_local(),
            A->get_npart_local());

  A->reset_version_tracker();
  A->remove_particles(aa);
  A->test_version_different();
  A->test_internal_state();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());
  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  ASSERT_EQ(bb->get_npart_local(), A->get_npart_local());
  ASSERT_EQ(aa->get_npart_local(), 0);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = id->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      ASSERT_TRUE(id->at(rowx, 0) % 2 == 1);
    }
    ASSERT_EQ(bb->get_npart_cell(cellx), A->get_npart_cell(cellx));
    ASSERT_EQ(aa->get_npart_cell(cellx), 0);
  }

  auto AA = std::make_shared<ParticleSubGroup>(A);

  A->reset_version_tracker();
  A->remove_particles(AA);
  A->test_version_different();
  A->test_init();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());
  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  ASSERT_EQ(A->get_npart_local(), 0);
  ASSERT_EQ(AA->get_npart_local(), 0);
  ASSERT_EQ(bb->get_npart_local(), 0);
  ASSERT_EQ(aa->get_npart_local(), 0);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(A->get_npart_cell(cellx), 0);
    ASSERT_EQ(AA->get_npart_cell(cellx), 0);
    ASSERT_EQ(bb->get_npart_cell(cellx), 0);
    ASSERT_EQ(aa->get_npart_cell(cellx), 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, clear) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto bb = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));
  auto ee = std::make_shared<ParticleSubGroup>(
      A, [=](auto /*ID*/) { return false; }, Access::read(Sym<INT>("ID")));

  auto AA = std::make_shared<ParticleSubGroup>(A);
  A->reset_version_tracker();
  A->clear();
  A->test_version_different();
  A->test_init();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());
  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  ASSERT_EQ(A->get_npart_local(), 0);
  ASSERT_EQ(AA->get_npart_local(), 0);
  ASSERT_EQ(bb->get_npart_local(), 0);
  ASSERT_EQ(aa->get_npart_local(), 0);
  ASSERT_EQ(ee->get_npart_local(), 0);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(A->get_npart_cell(cellx), 0);
    ASSERT_EQ(AA->get_npart_cell(cellx), 0);
    ASSERT_EQ(bb->get_npart_cell(cellx), 0);
    ASSERT_EQ(aa->get_npart_cell(cellx), 0);
    ASSERT_EQ(ee->get_npart_cell(cellx), 0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, add_product_matrix) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto bb = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));
  auto ee = std::make_shared<ParticleSubGroup>(
      A, [=](auto /*ID*/) { return false; }, Access::read(Sym<INT>("ID")));

  const int npart_b_A = A->get_npart_local();
  const int npart_b_aa = aa->get_npart_local();
  const int npart_b_bb = bb->get_npart_local();

  auto product_spec = product_matrix_spec(ParticleProp(Sym<INT>("MARKER"), 1));
  auto pm = product_matrix(sycl_target, product_spec);
  pm->reset(1);

  A->reset_version_tracker();
  A->add_particles_local(pm);
  A->test_version_different();
  A->test_internal_state();

  ASSERT_EQ(A->get_npart_local(), npart_b_A + 1);
  ASSERT_EQ(bb->get_npart_local(), npart_b_bb);
  ASSERT_EQ(aa->get_npart_local(), npart_b_aa + 1);
  ASSERT_EQ(ee->get_npart_local(), 0);

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, add_particles_local_particle_group) {
  auto A = particle_loop_common(10);
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto aa = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  auto B =
      make_test_obj<ParticleGroup>(domain, A->get_particle_spec(), sycl_target);

  auto product_spec = product_matrix_spec(ParticleProp(Sym<INT>("MARKER"), 1));
  auto pm = product_matrix(sycl_target, product_spec);
  pm->reset(1);

  B->reset_version_tracker();
  B->add_particles_local(pm);
  B->test_version_different();
  B->test_internal_state();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  A->reset_version_tracker();
  A->add_particles_local(B);
  A->test_version_different();
  A->test_internal_state();

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  B->free();
  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, add_particles_local_particle_sub_group) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto odd = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 1; }, Access::read(Sym<INT>("ID")));
  auto even = std::make_shared<ParticleSubGroup>(
      A, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  auto B =
      make_test_obj<ParticleGroup>(domain, A->get_particle_spec(), sycl_target);

  auto bb = std::make_shared<ParticleSubGroup>(
      B, [=](auto ID) { return ID[0] % 2 == 0; }, Access::read(Sym<INT>("ID")));

  B->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<INT>("FOO"), 3), cell_count));

  const int npart_local = A->get_npart_local();
  const int npart_local_even = even->get_npart_local();
  const int npart_local_odd = odd->get_npart_local();
  ASSERT_EQ(npart_local, npart_local_even + npart_local_odd);

  std::vector<int> npart_cell(cell_count);
  std::vector<int> npart_cell_even(cell_count);
  std::vector<int> npart_cell_odd(cell_count);

  std::map<int, std::set<INT>> cell_to_ids;
  std::set<INT> ids;

  for (int cx = 0; cx < cell_count; cx++) {
    npart_cell.at(cx) = A->get_npart_cell(cx);
    npart_cell_even.at(cx) = even->get_npart_cell(cx);
    npart_cell_odd.at(cx) = odd->get_npart_cell(cx);

    const int npart_A = A->get_npart_cell(cx);
    auto A_ID = A->get_cell(Sym<INT>("ID"), cx);
    for (int rowx = 0; rowx < npart_A; rowx++) {
      const INT id = A_ID->at(rowx, 0);
      cell_to_ids[cx].insert(id);
      ids.insert(id);
    }
    ASSERT_EQ(cell_to_ids[cx].size(), npart_cell.at(cx));
  }
  ASSERT_EQ(ids.size(), npart_local);

  B->reset_version_tracker();
  B->add_particles_local(even);
  B->test_version_different();
  B->test_internal_state();

  EXPECT_TRUE(bb->create_if_required());
  EXPECT_FALSE(bb->create_if_required());

  for (int cx = 0; cx < cell_count; cx++) {
    const int npart = B->get_npart_cell(cx);
    ASSERT_EQ(npart, npart_cell_even.at(cx));
    auto B_ID = B->get_cell(Sym<INT>("ID"), cx);
    for (int rowx = 0; rowx < npart; rowx++) {
      ASSERT_TRUE(B_ID->at(rowx, 0) % 2 == 0);
    }
  }

  A->reset_version_tracker();
  A->remove_particles(even);
  A->test_version_different();
  A->test_internal_state();

  EXPECT_TRUE(even->create_if_required());
  EXPECT_FALSE(even->create_if_required());
  ASSERT_EQ(even->get_npart_local(), 0);
  ASSERT_EQ(npart_local_odd, A->get_npart_local());
  ASSERT_EQ(npart_local_even, B->get_npart_local());

  std::set<INT> ids_to_test;
  for (int cx = 0; cx < cell_count; cx++) {
    const int npart_A = A->get_npart_cell(cx);
    const int npart_B = B->get_npart_cell(cx);

    ASSERT_EQ(npart_A, npart_cell_odd.at(cx));
    ASSERT_EQ(npart_B, npart_cell_even.at(cx));

    auto B_ID = B->get_cell(Sym<INT>("ID"), cx);
    auto A_ID = A->get_cell(Sym<INT>("ID"), cx);
    std::set<INT> cell_ids;

    for (int rowx = 0; rowx < npart_A; rowx++) {
      const INT id = A_ID->at(rowx, 0);
      ASSERT_TRUE(-1 < id);
      ASSERT_TRUE(id % 2 == 1);
      cell_ids.insert(id);
      ids_to_test.insert(id);
    }
    for (int rowx = 0; rowx < npart_B; rowx++) {
      const INT id = B_ID->at(rowx, 0);
      ASSERT_TRUE(-1 < id);
      ASSERT_TRUE(id % 2 == 0);
      cell_ids.insert(id);
      ids_to_test.insert(id);
    }
    ASSERT_TRUE(cell_ids == cell_to_ids[cx]);
  }
  ASSERT_EQ(ids_to_test.size(), ids.size());
  ASSERT_TRUE(ids_to_test == ids);

  for (int cx = 0; cx < cell_count; cx++) {
    const int npart = B->get_npart_cell(cx);
    auto FOO = B->get_cell(Sym<INT>("FOO"), cx);
    for (int rx = 0; rx < npart; rx++) {
      for (int dx = 0; dx < 3; dx++) {
        ASSERT_EQ(FOO->at(rx, dx), 0);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, sub_sub_group) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto mod2 = std::make_shared<TestParticleSubGroup>(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  auto mod4 = std::make_shared<TestParticleSubGroup>(
      std::dynamic_pointer_cast<ParticleSubGroup>(mod2),
      [=](auto ID) { return ((ID[0] % 4) == 0); },
      Access::read(Sym<INT>("ID")));

  ASSERT_EQ(mod2->get_particle_group(), mod4->get_particle_group());

  std::vector<INT> cells;
  std::vector<INT> layers;
  auto num_particles = mod2->test_get_cells_layers(cells, layers);
  std::set<std::pair<INT, INT>> correct, to_test;
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);
    auto id = A->get_cell(Sym<INT>("ID"), cellx)->at(layerx, 0);
    ASSERT_TRUE(id % 2 == 0);
    if (id % 4 == 0) {
      correct.insert({cellx, layerx});
    }
  }

  num_particles = mod4->test_get_cells_layers(cells, layers);
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);
    auto id = A->get_cell(Sym<INT>("ID"), cellx)->at(layerx, 0);
    ASSERT_TRUE(id % 4 == 0);
    to_test.insert({cellx, layerx});
  }
  ASSERT_EQ(to_test, correct);

  to_test.clear();
  auto AA = particle_sub_group(A);
  auto mod42 = std::make_shared<TestParticleSubGroup>(
      AA, [=](auto ID) { return ((ID[0] % 4) == 0); },
      Access::read(Sym<INT>("ID")));

  num_particles = mod42->test_get_cells_layers(cells, layers);
  for (int px = 0; px < num_particles; px++) {
    const int cellx = cells.at(px);
    const int layerx = layers.at(px);
    auto id = A->get_cell(Sym<INT>("ID"), cellx)->at(layerx, 0);
    ASSERT_TRUE(id % 4 == 0);
    to_test.insert({cellx, layerx});
  }
  ASSERT_EQ(to_test, correct);

  auto m0 = particle_sub_group(
      A, [=](auto /*M*/) { return true; }, Access::read(Sym<INT>("MARKER")));

  auto e0 = particle_sub_group(
      m0, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  EXPECT_TRUE(e0->create_if_required());
  EXPECT_FALSE(e0->create_if_required());

  particle_loop(
      A, [=](auto M) { M.at(0) += 1; }, Access::write(Sym<INT>("MARKER")))
      ->execute();

  EXPECT_TRUE(e0->create_if_required());
  EXPECT_FALSE(e0->create_if_required());

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, get_particles) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  const int cell_count = domain->mesh->get_cell_count();
  auto A = make_test_obj<ParticleGroup>(domain, particle_spec, sycl_target);

  A->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3), cell_count));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);

  const int N = cell_count * 2 + 7;

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % (cell_count - 1);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  A->add_particles_local(initial_distribution);
  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int npart_cell = aa->get_npart_cell(cellx);
    std::vector<INT> cells;
    std::vector<INT> layers;
    cells.reserve(npart_cell);
    layers.reserve(npart_cell);
    for (int layerx = 0; layerx < npart_cell; layerx++) {
      cells.push_back(cellx);
      layers.push_back(layerx);
    }
    auto particles = aa->get_particles(cells, layers);
    ASSERT_EQ(particles->npart, npart_cell);

    for (int layerx = 0; layerx < npart_cell; layerx++) {
      const INT px = particles->at(Sym<INT>("ID"), layerx, 0);
      EXPECT_EQ(px % (cell_count - 1), cellx);
      EXPECT_TRUE(px % 2 == 0);
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("P"), px, 0),
                particles->at(Sym<REAL>("P"), layerx, 0));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("P"), px, 1),
                particles->at(Sym<REAL>("P"), layerx, 1));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("V"), px, 0),
                particles->at(Sym<REAL>("V"), layerx, 0));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("V"), px, 1),
                particles->at(Sym<REAL>("V"), layerx, 1));
      EXPECT_EQ(initial_distribution.at(Sym<REAL>("V"), px, 2),
                particles->at(Sym<REAL>("V"), layerx, 2));

      EXPECT_EQ(particles->at(Sym<INT>("CELL_ID"), layerx, 0), cellx);
      EXPECT_EQ(particles->at(Sym<REAL>("FOO"), layerx, 0), 0.0);
      EXPECT_EQ(particles->at(Sym<REAL>("FOO"), layerx, 1), 0.0);
      EXPECT_EQ(particles->at(Sym<REAL>("FOO"), layerx, 2), 0.0);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, single_cell_base) {

  const int ndim = 2;
  std::vector<int> dims(ndim);
  dims[0] = 4;
  dims[1] = 8;

  const double cell_extent = 1.0;
  const int subdivision_order = 2;
  auto mesh = std::make_shared<CartesianHMesh>(MPI_COMM_WORLD, ndim, dims,
                                               cell_extent, subdivision_order);

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  const int cell_count = domain->mesh->get_cell_count();
  auto A = make_test_obj<ParticleGroup>(domain, particle_spec, sycl_target);

  A->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<REAL>("FOO"), 3), cell_count));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);

  const int N = cell_count * 3 + 7;

  auto positions =
      uniform_within_extents(N, ndim, mesh->global_extents, rng_pos);
  auto velocities =
      NESO::Particles::normal_distribution(N, 3, 0.0, 1.0, rng_vel);

  ParticleSet initial_distribution(N, particle_spec);

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] = positions[dimx][px];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % (cell_count - 1);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  A->add_particles_local(initial_distribution);

  const int cm1 = cell_count - 1;
  auto aam1 = particle_sub_group(A, cm1);

  ASSERT_TRUE(!aam1->is_entire_particle_group());
  ASSERT_EQ(aam1->get_particle_group(), A);
  for (int cx = 0; cx < cell_count; cx++) {
    if (cx != cm1) {
      ASSERT_EQ(aam1->get_npart_cell(cx), 0);
    } else {
      ASSERT_EQ(aam1->get_npart_cell(cx), A->get_npart_cell(cx));
    }
  }
  ASSERT_EQ(aam1->get_npart_local(), A->get_npart_cell(cm1));
  ASSERT_TRUE(aam1->is_valid());
  ASSERT_TRUE(!aam1->static_status());

  // sub group from sub group selecting different cells.
  if (0 != cm1) {
    auto aam10 = particle_sub_group(aam1, 0);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(aam10->get_npart_cell(cx), 0);
    }
    ASSERT_EQ(aam10->get_npart_local(), 0);
  }
  // sub group from sub group selecting same cell.
  auto aam1m1 = particle_sub_group(aam1, cm1);
  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(aam1m1->get_npart_cell(cx), aam1->get_npart_cell(cx));
  }
  ASSERT_EQ(aam1m1->get_npart_local(), A->get_npart_cell(cm1));

  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();

  particle_loop(
      aam1, [=](auto INDEX) { NESO_KERNEL_ASSERT(INDEX.cell == cm1, k_ep); },
      Access::read(ParticleLoopIndex{}))
      ->execute();
  ASSERT_TRUE(!ep->get_flag());

  particle_loop(
      aam1m1, [=](auto /*INDEX*/) { NESO_KERNEL_ASSERT(false, k_ep); },
      Access::read(ParticleLoopIndex{}))
      ->execute();
  ASSERT_TRUE(!ep->get_flag());

  if (cell_count >= 2) {
    // Get all the particles in the last cell with even ID and move them to
    // cell 0
    auto aae = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));
    auto aaem1 = particle_sub_group(aae, cm1);

    auto la = std::make_shared<LocalArray<int>>(sycl_target,
                                                aaem1->get_npart_cell(cm1));
    la->fill(0);
    particle_loop(
        aaem1,
        [=](auto ID, auto INDEX, auto /*LA*/) {
          NESO_KERNEL_ASSERT(INDEX.cell == cm1, k_ep);
          NESO_KERNEL_ASSERT(ID.at(0) % 2 == 0, k_ep);
          NESO_KERNEL_ASSERT(INDEX.get_loop_linear_index() ==
                                 INDEX.get_sub_linear_index(),
                             k_ep);
        },
        Access::read(Sym<INT>("ID")), Access::read(ParticleLoopIndex{}),
        Access::add(la))
        ->execute();
    ASSERT_TRUE(!ep->get_flag());

    auto aa0 = particle_sub_group(A, 0);
    A->remove_particles(aa0);
    ASSERT_TRUE(aa0->create_if_required());
    ASSERT_EQ(A->get_npart_cell(0), 0);

    particle_loop(
        aaem1, [=](auto CELL) { CELL.at(0) = 0; },
        Access::write(Sym<INT>("CELL_ID")))
        ->execute();
    A->cell_move();

    ASSERT_TRUE(aa0->create_if_required());
    ASSERT_TRUE(aaem1->create_if_required());

    particle_loop(
        aa0, [=](auto ID) { NESO_KERNEL_ASSERT(ID.at(0) % 2 == 0, k_ep); },
        Access::read(Sym<INT>("ID")))
        ->execute();
    ASSERT_TRUE(!ep->get_flag());

    particle_loop(
        aam1, [=](auto ID) { NESO_KERNEL_ASSERT(ID.at(0) % 2 == 1, k_ep); },
        Access::read(Sym<INT>("ID")))
        ->execute();
    ASSERT_TRUE(!ep->get_flag());
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleSubGroup, range_cell_base) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  A->free();
  sycl_target->free();
  mesh->free();
}
