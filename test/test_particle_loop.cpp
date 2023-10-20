#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

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

  const int cell_count = mesh->get_cell_count();

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
  const int size = sycl_target->comm_pair.size_parent;
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

TEST(ParticleLoop, base) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  ParticleLoop pl(
      A,
      [=](Access::ParticleDat::Write<REAL> P2,
          Access::ParticleDat::Read<REAL> P) {
        for (int dx = 0; dx < ndim; dx++) {
          P2[dx] = P[dx];
        }
      },
      Access::write(Sym<REAL>("P2")), Access::read(Sym<REAL>("P")));

  pl.execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(cellx);
    auto p2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(cellx);
    const int nrow = p->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {
        ASSERT_EQ((*p)[dimx][rowx], (*p2)[dimx][rowx]);
      }
    }
  }

  ParticleLoop particle_loop_auto(
      A,
      [=](auto ID, auto P, auto V) {
        ID[0] = 42;
        for (int dx = 0; dx < ndim; dx++) {
          P[dx] += V[dx];
        }
      },
      Access::write(Sym<INT>("ID")), Access::write(Sym<REAL>("P")),
      Access::read(Sym<REAL>("V")));

  particle_loop_auto.execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(cellx);
    auto p2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(cellx);
    auto v = A->get_dat(Sym<REAL>("V"))->cell_dat.get_cell(cellx);
    auto id = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = p->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      ASSERT_EQ((*id)[0][rowx], 42);
      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {
        ASSERT_TRUE(std::abs((*p)[dimx][rowx] - (*v)[dimx][rowx] -
                             (*p2)[dimx][rowx]) < 1.0e-10);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

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

  std::vector<int> d1(3);
  std::fill(d1.begin(), d1.end(), 0);
  LocalArray<int> l1(sycl_target, d1);

  ParticleLoop particle_loop_add(
      A,
      [=](Access::LocalArray::Add<int> L1) {
        L1(0, 1);
        L1(1, 2);
        L1(2, 3);
      },
      Access::add(l1));

  particle_loop_add.execute();
  l1.get(d1);
  EXPECT_EQ(d1[0], local_count);
  EXPECT_EQ(d1[1], local_count * 2);
  EXPECT_EQ(d1[2], local_count * 3);

  A->free();
  sycl_target->free();
  mesh->free();
}

namespace {

template <typename KERNEL> struct KernelTest {

  KERNEL kernel;
  KernelTest(KERNEL kernel) : kernel(kernel) {}

  template <typename... ARGS> inline void operator()(ARGS... args) {
    kernel(args...);
  }
};

} // namespace

TEST(ParticleLoop, templated_kernel) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  const int N = 3;
  std::vector<REAL> d0(N);
  std::iota(d0.begin(), d0.end(), 1);
  LocalArray<REAL> l0(sycl_target, d0);

  KernelTest k([=](Access::ParticleDat::Write<REAL> P2,
                   Access::LocalArray::Read<REAL> L0) {
    for (int dx = 0; dx < ndim; dx++) {
      P2[dx] = L0[dx];
    }
  });

  ParticleLoop pl(A, k, Access::write(Sym<REAL>("P2")), Access::read(l0));

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

  A->free();
  mesh->free();
}

TEST(ParticleLoop, base_pointer) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto pl = particle_loop(
      A,
      [=](Access::ParticleDat::Write<REAL> P2,
          Access::ParticleDat::Read<REAL> P) {
        for (int dx = 0; dx < ndim; dx++) {
          P2[dx] = P[dx];
        }
      },
      Access::write(Sym<REAL>("P2")), Access::read(Sym<REAL>("P")));

  pl->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto p = A->get_dat(Sym<REAL>("P"))->cell_dat.get_cell(cellx);
    auto p2 = A->get_dat(Sym<REAL>("P2"))->cell_dat.get_cell(cellx);
    const int nrow = p->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      for (int dimx = 0; dimx < ndim; dimx++) {
        ASSERT_EQ((*p)[dimx][rowx], (*p2)[dimx][rowx]);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoop, loop_index) {
  auto A = particle_loop_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  auto pl = particle_loop(
      A,
      [=](Access::LoopIndex::Read cell_layer,
          Access::ParticleDat::Write<INT> loop_index) {
        loop_index[0] = cell_layer.cell;
        loop_index[1] = cell_layer.layer;
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("LOOP_INDEX")));

  pl->execute();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto loop_index =
        A->get_dat(Sym<INT>("LOOP_INDEX"))->cell_dat.get_cell(cellx);
    const int nrow = loop_index->nrow;

    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      // for each dimension
      ASSERT_EQ((*loop_index)[0][rowx], cellx);
      ASSERT_EQ((*loop_index)[1][rowx], rowx);
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(ParticleLoop, global_array) {
  const int N_per_rank = 1093;
  auto A = particle_loop_common(N_per_rank);
  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  const int N = 3;
  GlobalArray<REAL> g0(sycl_target, N, 53);

  ParticleLoop pl(
      A,
      [=](Access::ParticleDat::Write<REAL> P2,
          Access::GlobalArray::Read<REAL> G0) {
        for (int dx = 0; dx < ndim; dx++) {
          P2[dx] = G0[dx];
        }
      },
      Access::write(Sym<REAL>("P2")), Access::read(g0));

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
        EXPECT_EQ((*p2)[dimx][rowx], 53);
      }
    }
  }

  GlobalArray<int> g1(sycl_target, N, 0);

  ParticleLoop pl_add(
      A,
      [=](Access::GlobalArray::Add<int> G1) {
        G1(0, 1);
        G1(1, 2);
        G1(2, 3);
      },
      Access::add(g1));

  pl_add.execute();

  auto d1 = g1.get();

  const int N_total = N_per_rank * size;
  ASSERT_EQ(d1.at(0), N_total);
  ASSERT_EQ(d1.at(1), N_total * 2);
  ASSERT_EQ(d1.at(2), N_total * 3);

  A->free();
  sycl_target->free();
  mesh->free();
}
