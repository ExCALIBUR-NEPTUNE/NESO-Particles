#include <gtest/gtest.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

TEST(ParticleSetDevice, test_particle_set_device) {

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), 2, true),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  const int N = 10;
  auto set_host = std::make_shared<ParticleSet>(N, particle_spec);
  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < 2; dimx++) {
      set_host->at(Sym<REAL>("P"), px, dimx) = (double)((px * 2) + (dimx));
    }
    set_host->at(Sym<INT>("ID"), px, 0) = px;
  }

  for (int px = 0; px < N; px++) {
    for (int dimx = 0; dimx < 2; dimx++) {
      ASSERT_EQ((double)((px * 2) + (dimx)),
                set_host->at(Sym<REAL>("P"), px, dimx));
    }
    ASSERT_EQ(px, set_host->at(Sym<INT>("ID"), px, 0));
  }

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  {
    ParticleSetDevice set_device(sycl_target, N, particle_spec);
    set_device.set(set_host);
    auto to_test = set_device.get();

    for (int px = 0; px < N; px++) {
      for (int cx = 0; cx < 2; cx++) {
        ASSERT_EQ(to_test->at(Sym<REAL>("P"), px, cx),
                  set_host->at(Sym<REAL>("P"), px, cx));
      }
      for (int cx = 0; cx < 3; cx++) {
        ASSERT_EQ(to_test->at(Sym<REAL>("V"), px, cx),
                  set_host->at(Sym<REAL>("V"), px, cx));
      }
      ASSERT_EQ(to_test->at(Sym<INT>("CELL_ID"), px, 0),
                set_host->at(Sym<INT>("CELL_ID"), px, 0));
      ASSERT_EQ(to_test->at(Sym<INT>("ID"), px, 0),
                set_host->at(Sym<INT>("ID"), px, 0));
    }
  }

  sycl_target->free();
}
