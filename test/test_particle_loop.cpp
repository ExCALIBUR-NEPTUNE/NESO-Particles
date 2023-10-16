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

  auto sycl_target =
      std::make_shared<SYCLTarget>(GPU_SELECTOR, mesh->get_comm());

  auto domain = std::make_shared<Domain>(mesh);

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<REAL>("FOO"), 3),
                                  domain->mesh->get_cell_count()));

  std::mt19937 rng_pos(52234234);
  std::mt19937 rng_vel(52234231);

  const int N = 10;

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

  // A->print(Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("ID"));

  ParticleLoop particle_loop(
      A,
      [=](Access::ParticleDat::Write<REAL> P, Access::ParticleDat::Read<REAL> V,
          Access::ParticleDat::Write<INT> ID) {
        P[0] += V[0];
        ID[0] = -42;
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")),
      Access::write(Sym<INT>("ID")));

  particle_loop.execute();

  A->print(Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("ID"));

  ParticleLoop particle_loop_auto(
      A,
      [=](auto P, auto V, auto ID) {
        P[0] += V[0];
        ID[0] = 43;
      },
      Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")),
      Access::write(Sym<INT>("ID")));

  particle_loop_auto.execute();

  A->print(Sym<REAL>("P"), Sym<REAL>("V"), Sym<INT>("ID"));

  A->free();
  mesh->free();
}
