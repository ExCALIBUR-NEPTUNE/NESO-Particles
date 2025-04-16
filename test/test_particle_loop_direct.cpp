#include "include/test_neso_particles.hpp"

TEST(ParticleLoopDirect, read_write_access) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  const REAL dt = 0.01;

  particle_loop(
      A,
      [=](auto P2, auto P, auto V) {
        P2.at(0) = P.at(0) + dt * V.at(0);
        P2.at(1) = P.at(1) + dt * V.at(1);
      },
      Access::write(Sym<REAL>("P2")), Access::read(Sym<REAL>("P")),
      Access::read(Sym<REAL>("V")))
      ->execute();

  auto d_V = Access::direct_get(Access::read(A->get_dat(Sym<REAL>("V"))));
  auto d_P = Access::direct_get(Access::write(A->get_dat(Sym<REAL>("P"))));

  ParticleLoopImplementation::ParticleLoopBlockIterationSet block_iteration_set(
      A->get_dat(Sym<REAL>("P")));

  auto iteration_set = block_iteration_set.get_all_cells();

  EventStack event_stack;
  for (auto &block : iteration_set) {
    const auto block_device = block.block_device;
    event_stack.push(sycl_target->queue.parallel_for<>(
        block.loop_iteration_set, [=](sycl::nd_item<2> idx) {
          std::size_t cell;
          std::size_t layer;
          block_device.get_cell_layer(idx, &cell, &layer);
          if (block_device.work_item_required(cell, layer)) {
            d_P[cell][0][layer] += dt * d_V[cell][0][layer];
            d_P[cell][1][layer] += dt * d_V[cell][1][layer];
          }
        }));
  }
  event_stack.wait();

  Access::direct_restore(Access::read(A->get_dat(Sym<REAL>("V"))), d_V);
  Access::direct_restore(Access::write(A->get_dat(Sym<REAL>("P"))), d_P);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  particle_loop(
      A,
      [=](auto P2, auto P) {
        NESO_KERNEL_ASSERT(Kernel::abs(P2.at(0) - P.at(0)) < 1.0e-14, k_ep);
        NESO_KERNEL_ASSERT(Kernel::abs(P2.at(1) - P.at(1)) < 1.0e-14, k_ep);
      },
      Access::read(Sym<REAL>("P2")), Access::read(Sym<REAL>("P")))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  sycl_target->free();
}

TEST(ParticleLoopDirect, cache_invalidation) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());

  {
    auto d_P = Access::direct_get(Access::write(A->get_dat(Sym<REAL>("P"))));
    Access::direct_restore(Access::write(A->get_dat(Sym<REAL>("P"))), d_P);
  }
  EXPECT_FALSE(aa->create_if_required());
  {
    auto d_ID = Access::direct_get(Access::write(A->get_dat(Sym<INT>("ID"))));
    Access::direct_restore(Access::write(A->get_dat(Sym<INT>("ID"))), d_ID);
  }
  EXPECT_TRUE(aa->create_if_required());
  EXPECT_FALSE(aa->create_if_required());
  {
    auto d_ID = Access::direct_get(Access::read(A->get_dat(Sym<INT>("ID"))));
    Access::direct_restore(Access::read(A->get_dat(Sym<INT>("ID"))), d_ID);
  }

  EXPECT_FALSE(aa->create_if_required());
  sycl_target->free();
}
