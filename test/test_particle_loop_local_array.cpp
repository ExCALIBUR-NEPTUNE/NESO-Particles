#include "include/test_neso_particles.hpp"

namespace {

const int ndim = 2;

ParticleGroupSharedPtr particle_loop_common(const int N = 1093) {
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
                             ParticleProp(Sym<INT>("LOOP_INDEX"), 2),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
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

} // namespace

TEST(ParticleLoop, local_array) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  const int N = 3;
  std::vector<REAL> d0(N);
  std::iota(d0.begin(), d0.end(), 1);
  LocalArray<REAL> l0(sycl_target, d0);

  ParticleLoop pl(
      A,
      [=](Access::ParticleDat::Write<REAL> P2,
          Access::LocalArray::Read<REAL> L0) {
        for (int dx = 0; dx < ndim; dx++) {
          P2[dx] = L0[dx];
        }
      },
      Access::write(Sym<REAL>("P2")), Access::read(l0));

  pl.execute();

  int local_count = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(cellx);
    const int nrow = p2->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      local_count++;
      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {
        EXPECT_EQ((*p2)[dimx][rowx], d0[dimx]);
      }
    }
  }

  auto l0ptr = std::make_shared<LocalArray<REAL>>(sycl_target, d0);
  auto plptr = particle_loop(
      A,
      [=](Access::ParticleDat::Write<REAL> P2,
          Access::LocalArray::Read<REAL> L0) {
        for (int dx = 0; dx < ndim; dx++) {
          P2[dx] = L0[dx] * 2.0;
        }
      },
      Access::write(Sym<REAL>("P2")), Access::read(l0ptr));

  plptr->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(cellx);
    const int nrow = p2->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {
        EXPECT_TRUE(std::abs((*p2)[dimx][rowx] - d0[dimx] * 2.0) < 1.0e-14);
      }
    }
  }

  std::vector<int> d1(3);
  std::fill(d1.begin(), d1.end(), 0);
  LocalArray<int> l1(sycl_target, d1);

  ParticleLoop particle_loop_add(
      A,
      [=](Access::LocalArray::Add<int> L1) {
        L1.fetch_add(0, 1);
        L1.fetch_add(1, 2);
        L1.fetch_add(2, 3);
      },
      Access::add(l1));

  particle_loop_add.execute();
  l1.get(d1);
  EXPECT_EQ(d1[0], local_count);
  EXPECT_EQ(d1[1], local_count * 2);
  EXPECT_EQ(d1[2], local_count * 3);

  // LocalArray write
  const int num_write = A->get_npart_local();
  std::vector<INT> h_law(num_write);
  std::fill(h_law.begin(), h_law.end(), 0);
  auto law = std::make_shared<LocalArray<INT>>(sycl_target, h_law);
  auto law_index = std::make_shared<LocalArray<INT>>(sycl_target, 1);
  law_index->fill(0);

  auto law_loop = particle_loop(
      A,
      [=](auto la_index, auto la) {
        const int index = la_index.fetch_add(0, 1);
        if (index % 2 == 0) {
          la[index] = 1;
        } else {
          la.at(index) = 1;
        }
      },
      Access::add(law_index), Access::write(law));
  law_loop->execute();

  auto post_law = law->get();
  for (int px = 0; px < num_write; px++) {
    EXPECT_EQ(post_law.at(px), 1);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(NDIndex, linear_index) {

  {
    NDIndex<1> index = {1};
    ASSERT_EQ(index.get_linear_index(0), 0);
  }
  {
    NDIndex<2> index = {3, 7};
    ASSERT_EQ(index.get_linear_index(0, 0), 0);
    ASSERT_EQ(index.get_linear_index(0, 1), 1);
    ASSERT_EQ(index.get_linear_index(0, 6), 6);
    ASSERT_EQ(index.get_linear_index(1, 0), 7);
    ASSERT_EQ(index.get_linear_index(1, 1), 8);
    ASSERT_EQ(index.get_linear_index(2, 6), 3 * 7 - 1);
  }
  {
    NDIndex<3> index = {3, 7, 5};
    ASSERT_EQ(index.get_linear_index(0, 0, 0), 0);
    ASSERT_EQ(index.get_linear_index(2, 6, 4), 3 * 7 * 5 - 1);
    ASSERT_EQ(index.get_linear_index(2, 6, 1), 3 * 7 * 5 - 4);
    ASSERT_EQ(index.size(), 3 * 7 * 5);
  }
}

TEST(ParticleLoop, nd_local_array) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto ndla = std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, 3, 5);
  ndla->fill(-1.0);

  auto h_ndla = ndla->get();
  ASSERT_EQ(h_ndla.size(), 3 * 5);

  for (int ix = 0; ix < 3 * 5; ix++) {
    ASSERT_EQ(h_ndla.at(ix), -1.0);
    h_ndla.at(ix) = 7.0;
  }

  ndla->set(h_ndla);
  std::fill(h_ndla.begin(), h_ndla.end(), -1.0);
  h_ndla = ndla->get();
  for (int ix = 0; ix < 3 * 5; ix++) {
    ASSERT_EQ(h_ndla.at(ix), 7.0);
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
