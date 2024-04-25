#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <random>

using namespace NESO::Particles;

TEST(ParticleSpec, test_particle_spec) {
  ParticleSpec a{ParticleProp(Sym<REAL>("P"), 2, true),
                 ParticleProp(Sym<REAL>("V"), 3),
                 ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                 ParticleProp(Sym<INT>("ID"), 1)};

  ParticleSpec b{ParticleProp(Sym<REAL>("P"), 2, true),
                 ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                 ParticleProp(Sym<INT>("ID"), 1)};

  ParticleSpec c{ParticleProp(Sym<REAL>("P"), 2, true),
                 ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                 ParticleProp(Sym<INT>("id"), 1)};

  ASSERT_TRUE(a.contains(ParticleProp(Sym<REAL>("P"), 2, true)));
  ASSERT_TRUE(a.contains(a));
  ASSERT_TRUE(a.contains(b));
  ASSERT_FALSE(a.contains(c));
}

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
