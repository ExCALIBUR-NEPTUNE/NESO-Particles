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

template <typename T, std::size_t N> struct Read {
  /// Pointer to underlying data for the array.
  Read() = default;
  T *ptr;
  NDIndex<N> index;
  template <typename... I> const T &at(I... ix) const {
    return ptr[index.get_linear_index(ix...)];
  }
};

TEST(ParticleLoop, nd_local_array_host) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  {
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
  }

  {
    auto ndla = std::make_shared<NDLocalArray<int, 2>>(sycl_target, 3, 5);
    ndla->fill(-1);

    auto h_ndla = ndla->get();
    ASSERT_EQ(h_ndla.size(), 3 * 5);

    for (int ix = 0; ix < 3 * 5; ix++) {
      ASSERT_EQ(h_ndla.at(ix), -1);
      h_ndla.at(ix) = 7;
    }

    ndla->set(h_ndla);
    std::fill(h_ndla.begin(), h_ndla.end(), -1);
    h_ndla = ndla->get();
    for (int ix = 0; ix < 3 * 5; ix++) {
      ASSERT_EQ(h_ndla.at(ix), 7);
    }
  }

  sycl_target->free();
}

TEST(ParticleLoop, nd_local_array_device) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  const int N = 3;
  const int M = 1;
  auto ndla =
      std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, cell_count, N, M);
  ndla->fill(0.0);

  auto d = ndla->get();
  int index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rx = 0; rx < N; rx++) {
      for (int cx = 0; cx < M; cx++) {
        d.at(cellx * N * M + rx * M + cx) = index++;
      }
    }
  }
  ndla->set(d);

  particle_loop(
      A,
      [=](auto index, auto V, auto LA) {
        V.at(0) = LA.at(index.cell, 0, 0);
        V.at(1) = LA.at(index.cell, 1, 0);
        V.at(2) = LA.at(index.cell, 2, 0);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<REAL>("V")),
      Access::read(ndla))
      ->execute();

  index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto V = A->get_dat(Sym<REAL>("V"))->cell_dat.get_cell(cellx);
    const int nrow = V->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      ASSERT_EQ(V->at(rowx, 0), (REAL)(index + 0));
      ASSERT_EQ(V->at(rowx, 1), (REAL)(index + 1));
      ASSERT_EQ(V->at(rowx, 2), (REAL)(index + 2));
    }
    index += M * N;
  }

  {
    auto ndla_int =
        std::make_shared<NDLocalArray<int, 1>>(sycl_target, cell_count);
    ndla_int->fill(0);

    particle_loop(
        A, [=](auto index, auto LA) { LA.fetch_add(index.cell, 1); },
        Access::read(ParticleLoopIndex{}), Access::add(ndla_int))
        ->execute();

    auto c = ndla_int->get();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      ASSERT_EQ(c.at(cellx), A->get_npart_cell(cellx));
    }
  }

  {
    auto ndla_int =
        std::make_shared<NDLocalArray<int, 2>>(sycl_target, cell_count, 1);
    ndla_int->fill(0);

    particle_loop(
        A, [=](auto index, auto LA) { LA.fetch_add(index.cell, 0, 1); },
        Access::read(ParticleLoopIndex{}), Access::add(ndla_int))
        ->execute();

    auto c = ndla_int->get();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      ASSERT_EQ(c.at(cellx), A->get_npart_cell(cellx));
    }
  }

  {
    INT max_npart_cell = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      max_npart_cell = std::max(max_npart_cell, A->get_npart_cell(cellx));
    }
    auto ndla_int = std::make_shared<NDLocalArray<int, 3>>(
        sycl_target, cell_count, max_npart_cell, 2);
    ndla_int->fill(-1);

    particle_loop(
        A,
        [=](auto index, auto LA) {
          LA.at(index.cell, index.layer, 0) = index.cell;
          LA.at(index.cell, index.layer, 1) = index.layer;
        },
        Access::read(ParticleLoopIndex{}), Access::write(ndla_int))
        ->execute();

    auto d = ndla_int->get();

    for (int cellx = 0; cellx < cell_count; cellx++) {
      const INT num_layers = A->get_npart_cell(cellx);
      for (int rx = 0; rx < num_layers; rx++) {
        ASSERT_EQ(d.at(cellx * max_npart_cell * 2 + rx * 2 + 0), cellx);
        ASSERT_EQ(d.at(cellx * max_npart_cell * 2 + rx * 2 + 1), rx);
      }
      for (int rx = num_layers; rx < max_npart_cell; rx++) {
        ASSERT_EQ(d.at(cellx * max_npart_cell * 2 + rx * 2 + 0), -1);
        ASSERT_EQ(d.at(cellx * max_npart_cell * 2 + rx * 2 + 1), -1);
      }
    }
  }

  {
    auto ndla_real =
        std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, cell_count, 3, 1);
    ndla_real->fill(0.0);
    auto cdc_real =
        std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, 3, 1);
    cdc_real->fill(0.0);

    particle_loop(
        A,
        [=](auto index, auto V, auto LA, auto CDC) {
          LA.fetch_add(index.cell, 0, 0, V.at(0));
          LA.fetch_add(index.cell, 1, 0, V.at(1));
          LA.fetch_add(index.cell, 2, 0, V.at(2));
          CDC.fetch_add(0, 0, V.at(0));
          CDC.fetch_add(1, 0, V.at(1));
          CDC.fetch_add(2, 0, V.at(2));
        },
        Access::read(ParticleLoopIndex{}), Access::read(Sym<REAL>("V")),
        Access::add(ndla_real), Access::add(cdc_real))
        ->execute();

    auto h_ndla_real = ndla_real->get();

    int index = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      for (int rx = 0; rx < 3; rx++) {
        for (int cx = 0; cx < 1; cx++) {
          const REAL correct = cdc_real->get_value(cellx, rx, cx);
          const REAL to_test = h_ndla_real.at(index);
          ASSERT_NEAR(to_test, correct, 1.0e-15);
          index++;
        }
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
