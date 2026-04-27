#include "include/test_neso_particles.hpp"

TEST(ReductionContextCellwiseBins, base) {
  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_2d(511, 16, 32);
  auto A = A_t;
  const int cell_count = A->domain->mesh->get_cell_count();

  A->add_particle_dat(Sym<INT>("BIN"), 1);
  A->add_particle_dat(Sym<INT>("FOO"), 3);

  const int max_num_bins = 100;
  auto sycl_target = sycl_target_t;

  auto lambda_test = [&](auto g) {
    particle_loop(
        A,
        [=](auto FOO) {
          FOO.at(0) = -1;
          FOO.at(1) = -1;
          FOO.at(2) = -1;
        },
        Access::write(Sym<INT>("FOO")))
        ->execute();

    particle_loop(
        g,
        [=](auto INDEX, auto BIN) { BIN.at(0) = INDEX.layer % max_num_bins; },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("BIN")))
        ->execute();

    auto partition = get_index_map<2, 1>(sycl_target);
    partition_mesh_cells_bins(g, max_num_bins, Sym<INT>("BIN"), 0, partition);
    auto reduction_context =
        std::make_shared<ReductionContextCellwiseBins>(g, partition);

    auto cdc = std::make_shared<CellDatConst<int>>(sycl_target, cell_count,
                                                   max_num_bins, 1);

    ErrorPropagate ep(sycl_target);
    auto k_ep = ep.device_ptr();

    { // Do plain kernels that access ParticleDats work?
      particle_loop(
          g,
          [=](auto INDEX, auto FOO) {
            FOO.at(0) = INDEX.layer;
            FOO.at(2) = 1;
          },
          Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("FOO")))
          ->execute();
      particle_loop(
          reduction_context,
          [=](auto INDEX, auto FOO) {
            NESO_KERNEL_ASSERT(INDEX.layer == FOO.at(0), k_ep);
            FOO.at(1) = FOO.at(0);
          },
          Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("FOO")))
          ->execute();
      ASSERT_FALSE(ep.get_flag());
      particle_loop(
          A,
          [=](auto FOO) {
            if (FOO.at(2) < 0) {
              NESO_KERNEL_ASSERT(FOO.at(0) < 0, k_ep);
              NESO_KERNEL_ASSERT(FOO.at(1) < 0, k_ep);
            } else {
              NESO_KERNEL_ASSERT(FOO.at(0) >= 0, k_ep);
              NESO_KERNEL_ASSERT(FOO.at(1) >= 0, k_ep);
            }
            NESO_KERNEL_ASSERT(FOO.at(0) == FOO.at(1), k_ep);
          },
          Access::read(Sym<INT>("FOO")))
          ->execute();
      ASSERT_FALSE(ep.get_flag());
    }

    {
      particle_loop(
          reduction_context,
          [=](auto INDEX, auto FOO, auto CDC) {
            FOO.at(0) = INDEX.layer;
            FOO.at(2) = 1;
          },
          Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("FOO")),
          Access::reduce(cdc, Kernel::plus<int>())
          )
          ->execute();


    }

    reduction_context->free();
    restore_index_map(sycl_target, partition);
  };

  lambda_test(A);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));
  lambda_test(aa);

  sycl_target_t->free();
  A->domain->mesh->free();
}
