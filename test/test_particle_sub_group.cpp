#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common() {
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

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  const int N = 1093; // prime
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
      A, [](auto ID, auto V) { V[0] += 0.0001; }, Access::read(Sym<INT>("ID")),
      Access::write(Sym<REAL>("V")));
  p0->execute();
  EXPECT_FALSE(aa->create_if_required());

  auto p1 = particle_loop(
      aa, [](auto ID, auto V) { V[0] += 0.0001; }, Access::read(Sym<INT>("ID")),
      Access::write(Sym<REAL>("V")));
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
    for (int rowx = 0; rowx < nrow; rowx++) {
      const INT mx = (*marker)[0][rowx];
      const INT ix = (*id)[0][rowx];
      if (ix % 2 == 0) {
        ASSERT_EQ(mx, index);
        index++;
      } else {
        ASSERT_EQ(mx, -1);
      }
    }
  }

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

  A->free();
  sycl_target->free();
  mesh->free();
}
