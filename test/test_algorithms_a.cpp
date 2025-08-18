#include "include/test_neso_particles.hpp"

namespace {
template <typename GROUP_TYPE, typename OP_TYPE_REAL, typename OP_TYPE_INT>
void wrapper_reduce_dat_component_cellwise(std::shared_ptr<GROUP_TYPE> A,
                                           SYCLTargetSharedPtr sycl_target,
                                           const int cell_count_t,
                                           const int mask, OP_TYPE_REAL op_real,
                                           OP_TYPE_INT op_int) {
  auto Vto_test =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  auto Vcorrect =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  Vto_test->fill(4.0);
  Vcorrect->fill(4.0);

  auto particle_group = get_particle_group(A);

  particle_loop(
      A,
      [=](auto V, auto VCDC) {
        for (int dx = 0; dx < 3; dx++) {
          VCDC.combine(dx, 0, V.at(dx));
        }
      },
      Access::read(Sym<REAL>("V")), Access::reduce(Vcorrect, op_real))
      ->execute();

  for (int dx = 0; dx < 3; dx++) {
    reduce_dat_component_cellwise(A, Sym<REAL>("V"), dx, Vto_test, dx, 0,
                                  op_real);
  }

  {
    auto correct = Vcorrect->get_all_cells();
    auto to_test = Vto_test->get_all_cells();

    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      auto V = particle_group->get_cell(Sym<REAL>("V"), cellx);
      auto ID = particle_group->get_cell(Sym<INT>("ID"), cellx);
      const auto nrow = V->nrow;
      for (int dx = 0; dx < 3; dx++) {
        REAL value = 4.0;

        for (int rx = 0; rx < nrow; rx++) {
          if (ID->at(rx, 0) % mask == 0) {
            value = op_real(value, V->at(rx, dx));
          }
        }

        ASSERT_TRUE(relative_error(correct.at(cellx)->at(dx, 0), value) <
                    1.0e-8);
        ASSERT_TRUE(relative_error(correct.at(cellx)->at(dx, 0),
                                   to_test.at(cellx)->at(dx, 0)) < 1.0e-8);
      }
    }
  }

  auto IDto_test =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 1, 1);
  auto IDcorrect =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 1, 1);
  Vto_test->fill(2);
  Vcorrect->fill(2);

  particle_loop(
      A, [=](auto ID, auto IDCDC) { IDCDC.combine(0, 0, ID.at(0)); },
      Access::read(Sym<INT>("ID")), Access::reduce(IDcorrect, op_int))
      ->execute();

  reduce_dat_component_cellwise(A, Sym<INT>("ID"), 0, IDto_test, 0, 0, op_int);

  {
    auto correct = IDcorrect->get_all_cells();
    auto to_test = IDto_test->get_all_cells();

    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      ASSERT_EQ(correct.at(cellx)->at(0, 0), to_test.at(cellx)->at(0, 0));
    }
  }
}

} // namespace

TEST(Algorithms, reduce_dat_component_cellwise) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  wrapper_reduce_dat_component_cellwise(A, sycl_target, cell_count_t, 1,
                                        Kernel::plus<REAL>(),
                                        Kernel::plus<INT>());
  wrapper_reduce_dat_component_cellwise(A, sycl_target, cell_count_t, 1,
                                        Kernel::minimum<REAL>(),
                                        Kernel::minimum<INT>());
  wrapper_reduce_dat_component_cellwise(A, sycl_target, cell_count_t, 1,
                                        Kernel::maximum<REAL>(),
                                        Kernel::maximum<INT>());

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, reduce_dat_component_cellwise_sub_group) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  wrapper_reduce_dat_component_cellwise(aa, sycl_target, cell_count_t, 2,
                                        Kernel::plus<REAL>(),
                                        Kernel::plus<INT>());
  wrapper_reduce_dat_component_cellwise(aa, sycl_target, cell_count_t, 2,
                                        Kernel::minimum<REAL>(),
                                        Kernel::minimum<INT>());
  wrapper_reduce_dat_component_cellwise(aa, sycl_target, cell_count_t, 2,
                                        Kernel::maximum<REAL>(),
                                        Kernel::maximum<INT>());
  wrapper_reduce_dat_component_cellwise(particle_sub_group(A), sycl_target,
                                        cell_count_t, 1, Kernel::plus<REAL>(),
                                        Kernel::plus<INT>());
  wrapper_reduce_dat_component_cellwise(
      particle_sub_group(A), sycl_target, cell_count_t, 1,
      Kernel::minimum<REAL>(), Kernel::minimum<INT>());
  wrapper_reduce_dat_component_cellwise(
      particle_sub_group(A), sycl_target, cell_count_t, 1,
      Kernel::maximum<REAL>(), Kernel::maximum<INT>());

  sycl_target->free();
  A->domain->mesh->free();
}
