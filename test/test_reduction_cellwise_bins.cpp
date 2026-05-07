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

      auto cdc_to_test = std::make_shared<CellDatConst<int>>(
          sycl_target, cell_count, max_num_bins, 1);

      auto cdc_correct = std::make_shared<CellDatConst<int>>(
          sycl_target, cell_count, max_num_bins, 1);

      cdc_to_test->fill(0);
      particle_loop(
          reduction_context, [=](auto CDC) { CDC.combine(0, 0, 1); },
          Access::reduce(cdc_to_test, Kernel::plus<int>()))
          ->execute();

      cdc_correct->fill(0);
      particle_loop(
          g, [=](auto BIN, auto CDC) { CDC.fetch_add(BIN.at(0), 0, 1); },
          Access::read(Sym<INT>("BIN")), Access::add(cdc_correct))
          ->execute();

      auto h_to_test = cdc_to_test->get_all_cells();
      auto h_correct = cdc_to_test->get_all_cells();

      for (int cellx = 0; cellx < cell_count; cellx++) {
        for (int binx = 0; binx < max_num_bins; binx++) {
          const auto correct = h_correct.at(cellx)->at(binx, 0);
          const auto to_test = h_to_test.at(cellx)->at(binx, 0);
          ASSERT_EQ(correct, to_test);
        }
      }
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

namespace {

template <typename T, typename OP>
void reduction_wrapper(const int num_components, OP op) {
  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_2d(511, 16, 32);
  auto A = A_t;
  const int cell_count = A->domain->mesh->get_cell_count();

  A->add_particle_dat(Sym<INT>("BIN"), 1);
  A->add_particle_dat(Sym<INT>("FOO"), 3);

  const int max_num_bins = 36;
  auto sycl_target = sycl_target_t;

  auto lambda_test = [&](auto g) {
    particle_loop(
        g,
        [=](auto INDEX, auto BIN) { BIN.at(0) = INDEX.layer % max_num_bins; },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("BIN")))
        ->execute();

    auto partition = get_index_map<2, 1>(sycl_target);
    partition_mesh_cells_bins(g, max_num_bins, Sym<INT>("BIN"), 0, partition);
    auto reduction_context =
        std::make_shared<ReductionContextCellwiseBins>(g, partition);

    for (int cx = 1; cx < (num_components + 1); cx++) {
      auto cdc_to_test = std::make_shared<CellDatConst<T>>(
          sycl_target, cell_count, max_num_bins, cx);

      auto cdc_correct = std::make_shared<CellDatConst<T>>(
          sycl_target, cell_count, max_num_bins, cx);

      cdc_to_test->fill(0);
      particle_loop(
          reduction_context,
          [=](auto CDC) {
            for (int dx = 0; dx < cx; dx++) {
              CDC.combine(0, dx, (T)dx + 1);
            }
          },
          Access::reduce(cdc_to_test, op))
          ->execute();

      cdc_correct->fill(0);
      particle_loop(
          g,
          [=](auto BIN, auto CDC) {
            for (int dx = 0; dx < cx; dx++) {
              CDC.combine(BIN.at(0), dx, (T)dx + 1);
            }
          },
          Access::read(Sym<INT>("BIN")), Access::reduce(cdc_correct, op))
          ->execute();

      auto h_to_test = cdc_to_test->get_all_cells();
      auto h_correct = cdc_to_test->get_all_cells();

      for (int cellx = 0; cellx < cell_count; cellx++) {
        for (int binx = 0; binx < max_num_bins; binx++) {
          const auto correct = h_correct.at(cellx)->at(binx, 0);
          const auto to_test = h_to_test.at(cellx)->at(binx, 0);

          if constexpr (std::is_same<T, int>::value) {
            ASSERT_EQ(correct, to_test);
          } else if constexpr (std::is_same<T, INT>::value) {
            ASSERT_EQ(correct, to_test);
          } else {
            const T error = relative_error(correct, to_test);
            ASSERT_TRUE(error < 1.0e-8);
          }
        }
      }
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

} // namespace

TEST(ReductionContextCellwiseBins, dims_types) {
  //reduction_wrapper<int>(5, Kernel::plus<int>());
  reduction_wrapper<REAL>(5, Kernel::plus<REAL>());
  //reduction_wrapper<INT>(5, Kernel::plus<INT>());
  //
  //reduction_wrapper<int>(3, Kernel::minimum<int>());
  //reduction_wrapper<REAL>(3, Kernel::minimum<REAL>());
  //reduction_wrapper<INT>(3, Kernel::minimum<INT>());
  //
  //reduction_wrapper<int>(1, Kernel::maximum<int>());
  //reduction_wrapper<REAL>(1, Kernel::maximum<REAL>());
  //reduction_wrapper<INT>(1, Kernel::maximum<INT>());
}
