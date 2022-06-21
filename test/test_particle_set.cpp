#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

TEST(ParticleSet, test_particle_set) {

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), 2, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  const int N = 10;
  ParticleSet initial_distribution(N, particle_spec);
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < 2; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          (double)((px * 2) + (dimx));
    }
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < 2; dimx++) {
      ASSERT_EQ((double)((px * 2) + (dimx)),
                initial_distribution[Sym<REAL>("P")][px][dimx]);
    }
    ASSERT_EQ(px, initial_distribution[Sym<INT>("ID")][px][0]);
  }
}
