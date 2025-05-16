#include "include/test_neso_particles.hpp"

using namespace NESO::Particles;

TEST(ParticleSetDevice, test_particle_set_device_base) {

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

TEST(ParticleSetDevice, test_particle_set_device_loop) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  ParticleSpec particle_spec{
      ParticleProp(Sym<INT>("A"), 2),
      ParticleProp(Sym<REAL>("B"), 3),
      ParticleProp(Sym<INT>("C"), 1),
      ParticleProp(Sym<REAL>("D"), 1),
  };

  const int npart_local = A->get_npart_local();
  auto set_device = std::make_shared<ParticleSetDevice>(
      sycl_target, npart_local, particle_spec);

  particle_loop(
      A,
      [=](auto INDEX, auto PSD, auto V) {
        PSD.at_int(INDEX.get_local_linear_index(), 0, 0) =
            INDEX.get_local_linear_index();
        PSD.at_int(INDEX.get_local_linear_index(), 0, 1) =
            INDEX.get_local_linear_index() - 100;
        for (int dx = 0; dx < 3; dx++) {
          PSD.at_real(INDEX.get_local_linear_index(), 0, dx) = V.at(dx);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(set_device),
      Access::read(Sym<REAL>("V")))
      ->execute();

  particle_loop(
      A,
      [=](auto PSD) {
        for (int nx = 0; nx < npart_local; nx++) {
          PSD.fetch_add_int(nx, 1, 0, 1);
          PSD.fetch_add_real(nx, 1, 0, 1.1);
        }
      },
      Access::add(set_device))
      ->execute();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  ASSERT_EQ(A->get_dat(Sym<REAL>("V"))->ncomp, 3);

  particle_loop(
      A,
      [=](auto INDEX, auto PSD, auto V) {
        NESO_KERNEL_ASSERT(PSD.at_int(INDEX.get_local_linear_index(), 0, 0) ==
                               INDEX.get_local_linear_index(),
                           k_ep);
        NESO_KERNEL_ASSERT(PSD.at_int(INDEX.get_local_linear_index(), 0, 1) ==
                               INDEX.get_local_linear_index() - 100,
                           k_ep);
        for (int dx = 0; dx < 3; dx++) {
          NESO_KERNEL_ASSERT(
              PSD.at_real(INDEX.get_local_linear_index(), 0, dx) == V.at(dx),
              k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(set_device),
      Access::read(Sym<REAL>("V")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  particle_loop(
      A,
      [=](auto INDEX, auto PSD) {
        NESO_KERNEL_ASSERT(PSD.at_int(INDEX.get_local_linear_index(), 1, 0) ==
                               npart_local,
                           k_ep);
        NESO_KERNEL_ASSERT(
            Kernel::abs(PSD.at_real(INDEX.get_local_linear_index(), 1, 0) -
                        npart_local * 1.1) /
                    (npart_local * 1.1) <
                1.0e-10,
            k_ep);
      },
      Access::read(ParticleLoopIndex{}), Access::read(set_device))
      ->execute();
  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
}
