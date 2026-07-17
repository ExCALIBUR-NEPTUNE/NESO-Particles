#include "include/test_neso_particles.hpp"

TEST(ParticleMask, base) {

  const int npart_cell = 25;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;

  A->add_particle_dat(Sym<INT>("FLAG"), 2);

  auto particle_mask = std::make_shared<ParticleMask>(sycl_target);

  ASSERT_EQ(particle_mask->sycl_target, sycl_target);
  ASSERT_EQ(particle_mask->num_masks_per_entry, 1);
  ASSERT_EQ(particle_mask->num_base_elements_per_entry, 1);
  ASSERT_EQ(particle_mask->size, 0);
  ASSERT_EQ(particle_mask->get_num_masks_true(), 0);

  particle_mask->reset(A);
  ASSERT_EQ(particle_mask->get_num_masks_true(), 0);

  ASSERT_EQ(particle_mask->size,
            static_cast<std::size_t>(A->get_npart_local()));

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      A,
      [=](auto INDEX, auto PARTICLE_MASK) {
        NESO_KERNEL_ASSERT(PARTICLE_MASK.get(INDEX) == false, k_ep);
      },
      Access::read(ParticleLoopIndex{}), Access::read(particle_mask))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  particle_loop(
      A,
      [=](auto INDEX, auto PARTICLE_MASK) {
        NESO_KERNEL_ASSERT(PARTICLE_MASK.get(INDEX) == false, k_ep);
        PARTICLE_MASK.set_on(INDEX);
        NESO_KERNEL_ASSERT(PARTICLE_MASK.get(INDEX) == true, k_ep);
        PARTICLE_MASK.set_off(INDEX);
        NESO_KERNEL_ASSERT(PARTICLE_MASK.get(INDEX) == false, k_ep);
        PARTICLE_MASK.set(INDEX, true);
        NESO_KERNEL_ASSERT(PARTICLE_MASK.get(INDEX) == true, k_ep);
      },
      Access::read(ParticleLoopIndex{}), Access::write(particle_mask))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  ASSERT_EQ(particle_mask->get_num_masks_true(),
            static_cast<std::size_t>(A->get_npart_local()));

  particle_loop(
      A, [=](auto FLAG) { FLAG.at(1) = 0; }, Access::write(Sym<INT>("FLAG")))
      ->execute();

  particle_mask->set(A, Sym<INT>("FLAG"), 1);
  ASSERT_EQ(particle_mask->get_num_masks_true(), 0);

  particle_loop(
      A, [=](auto FLAG) { FLAG.at(1) = 1; }, Access::write(Sym<INT>("FLAG")))
      ->execute();

  particle_mask->set(A, Sym<INT>("FLAG"), 1);

  ASSERT_EQ(particle_mask->get_num_masks_true(),
            static_cast<std::size_t>(A->get_npart_local()));

  particle_mask->reset(A);
  particle_mask->get(A, Sym<INT>("FLAG"), 1);
  particle_loop(
      A,
      [=](auto FLAG) {
        NESO_KERNEL_ASSERT(FLAG.at(1) == 0, k_ep);
        FLAG.at(1) = 1;
      },
      Access::write(Sym<INT>("FLAG")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  particle_mask->set(A, Sym<INT>("FLAG"), 1);

  ASSERT_EQ(particle_mask->get_num_masks_true(),
            static_cast<std::size_t>(A->get_npart_local()));

  particle_mask->set(A, false);

  ASSERT_EQ(particle_mask->get_num_masks_true(), 0);

  particle_mask->set(A, true);
  ASSERT_EQ(particle_mask->get_num_masks_true(),
            static_cast<std::size_t>(A->get_npart_local()));

  sycl_target->free();
  A->domain->mesh->free();
}
