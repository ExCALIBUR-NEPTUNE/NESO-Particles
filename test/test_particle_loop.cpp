#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

TEST(ParticleLoop, Base2) {

  const int ndim = 2;
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

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<REAL>("P2"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);

  const int N = 7901; // prime

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
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  A->add_particles_local(initial_distribution);
  parallel_advection_initialisation(A, 16);

  ParticleLoop particle_loop(
      A,
      [=](Access::ParticleDat::Write<REAL> P2,
          Access::ParticleDat::Read<REAL> P) {
        for (int dx = 0; dx < ndim; dx++) {
          P2[dx] = P[dx];
        }
      },
      Access::write(Sym<REAL>("P2")), Access::read(Sym<REAL>("P")));

  particle_loop.execute();

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
  mesh->free();
}
