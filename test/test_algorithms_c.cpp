#include "include/test_neso_particles.hpp"

TEST(Algorithms, copy_ephemeral_dat_to_particle_dat) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  A->add_particle_dat(Sym<INT>("FOO"), 4);
  A->add_particle_dat(Sym<INT>("BAR"), 4);
  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2; }, Access::read(Sym<INT>("ID")));

  aa->add_ephemeral_dat(Sym<INT>("FOO"), 4);

  particle_loop(
      aa,
      [=](auto INDEX, auto FOO) {
        for (int dx = 0; dx < 4; dx++) {
          FOO.at_ephemeral(dx) = (INDEX.get_local_linear_index() + dx * 7) % 31;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("FOO")))
      ->execute();

  copy_ephemeral_dat_to_particle_dat(aa, Sym<INT>("FOO"), Sym<INT>("FOO"));

  auto ep = ErrorPropagate(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      A,
      [=](auto INDEX, auto ID, auto FOO) {
        if (ID.at(0) % 2) {
          for (int dx = 0; dx < 4; dx++) {
            NESO_KERNEL_ASSERT(
                FOO.at(dx) == (INDEX.get_local_linear_index() + dx * 7) % 31,
                k_ep);
          }
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("ID")),
      Access::read(Sym<INT>("FOO")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  copy_ephemeral_dat_to_particle_dat(aa, Sym<INT>("FOO"), Sym<INT>("BAR"));

  particle_loop(
      A,
      [=](auto INDEX, auto ID, auto BAR) {
        if (ID.at(0) % 2) {
          for (int dx = 0; dx < 4; dx++) {
            NESO_KERNEL_ASSERT(
                BAR.at(dx) == (INDEX.get_local_linear_index() + dx * 7) % 31,
                k_ep);
          }
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("ID")),
      Access::read(Sym<INT>("BAR")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, copy_particle_dat_to_ephemeral_dat) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  A->add_particle_dat(Sym<INT>("FOO"), 4);
  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2; }, Access::read(Sym<INT>("ID")));
  aa->add_ephemeral_dat(Sym<INT>("FOO"), 4);
  aa->add_ephemeral_dat(Sym<INT>("BAR"), 4);

  particle_loop(
      A,
      [=](auto INDEX, auto FOO) {
        for (int dx = 0; dx < 4; dx++) {
          FOO.at(dx) = (INDEX.get_local_linear_index() + dx * 7) % 31;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("FOO")))
      ->execute();

  copy_particle_dat_to_ephemeral_dat(aa, Sym<INT>("FOO"), Sym<INT>("FOO"));

  auto ep = ErrorPropagate(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      aa,
      [=](auto INDEX, auto FOO) {
        for (int dx = 0; dx < 4; dx++) {
          NESO_KERNEL_ASSERT(FOO.at_ephemeral(dx) ==
                                 (INDEX.get_local_linear_index() + dx * 7) % 31,
                             k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("FOO")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  copy_particle_dat_to_ephemeral_dat(aa, Sym<INT>("FOO"), Sym<INT>("BAR"));

  particle_loop(
      aa,
      [=](auto INDEX, auto BAR) {
        for (int dx = 0; dx < 4; dx++) {
          NESO_KERNEL_ASSERT(BAR.at_ephemeral(dx) ==
                                 (INDEX.get_local_linear_index() + dx * 7) % 31,
                             k_ep);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(Sym<INT>("BAR")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
  A->domain->mesh->free();
}
