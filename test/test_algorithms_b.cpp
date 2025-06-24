#include "include/test_neso_particles.hpp"

namespace {

template <typename GROUP_TYPE, typename OP_TYPE_REAL, typename OP_TYPE_INT>
void wrapper_reduce_dat_components_cellwise(std::shared_ptr<GROUP_TYPE> A,
                                            SYCLTargetSharedPtr sycl_target,
                                            const int cell_count_t,
                                            const int mask,
                                            OP_TYPE_REAL op_real,
                                            OP_TYPE_INT op_int) {

  auto particle_group = get_particle_group(A);
  auto Vto_test =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  auto Vcorrect =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  Vto_test->fill(3.0);
  Vcorrect->fill(3.0);

  particle_loop(
      A,
      [=](auto V, auto VCDC) {
        for (int dx = 0; dx < 3; dx++) {
          VCDC.combine(dx, 0, V.at(dx));
        }
      },
      Access::read(Sym<REAL>("V")), Access::reduce(Vcorrect, op_real))
      ->execute();

  reduce_dat_components_cellwise(A, Sym<REAL>("V"), Vto_test, op_real);

  {
    auto correct = Vcorrect->get_all_cells();
    auto to_test = Vto_test->get_all_cells();

    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      auto V = particle_group->get_cell(Sym<REAL>("V"), cellx);
      auto ID = particle_group->get_cell(Sym<INT>("ID"), cellx);
      for (int dx = 0; dx < 3; dx++) {

        REAL value = 3.0;
        const auto nrow = V->nrow;

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

  auto FOOto_test =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 2, 2);
  auto FOOcorrect =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 2, 2);
  FOOto_test->fill(3);
  FOOcorrect->fill(3);

  particle_loop(
      A,
      [=](auto V, auto FOO) {
        FOO.at(0) = 4.0 * V.at(0);
        FOO.at(1) = 1.0 * V.at(1) + 2;
        FOO.at(2) = 2.0 * V.at(2) + 3;
        FOO.at(3) = 3.0 * V.at(0) + 1;
      },
      Access::read(Sym<REAL>("V")), Access::write(Sym<INT>("FOO")))
      ->execute();

  particle_loop(
      A,
      [=](auto FOO, auto FOOCDC) {
        FOOCDC.combine(0, 0, FOO.at(0));
        FOOCDC.combine(0, 1, FOO.at(1));
        FOOCDC.combine(1, 0, FOO.at(2));
        FOOCDC.combine(1, 1, FOO.at(3));
      },
      Access::read(Sym<INT>("FOO")), Access::reduce(FOOcorrect, op_int))
      ->execute();

  reduce_dat_components_cellwise(A, Sym<INT>("FOO"), FOOto_test, op_int);

  {
    auto correct = FOOcorrect->get_all_cells();
    auto to_test = FOOto_test->get_all_cells();

    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      auto FOO = particle_group->get_cell(Sym<INT>("FOO"), cellx);
      auto ID = particle_group->get_cell(Sym<INT>("ID"), cellx);

      int index = 0;
      for (int rx = 0; rx < 2; rx++) {
        for (int cx = 0; cx < 2; cx++) {

          INT value = 3;
          const auto nrow = FOO->nrow;
          for (int px = 0; px < nrow; px++) {
            if (ID->at(px, 0) % mask == 0) {
              value = op_int(value, FOO->at(px, index));
            }
          }
          ASSERT_EQ(correct.at(cellx)->at(rx, cx), value);
          ASSERT_EQ(correct.at(cellx)->at(rx, cx),
                    to_test.at(cellx)->at(rx, cx));

          index++;
        }
      }
    }
  }
}
} // namespace

TEST(Algorithms, reduce_dat_components_cellwise) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  A->add_particle_dat(Sym<INT>("FOO"), 4);
  wrapper_reduce_dat_components_cellwise(A, sycl_target, cell_count_t, 1,
                                         Kernel::plus<REAL>(),
                                         Kernel::plus<INT>());
  wrapper_reduce_dat_components_cellwise(A, sycl_target, cell_count_t, 1,
                                         Kernel::minimum<REAL>(),
                                         Kernel::minimum<INT>());
  wrapper_reduce_dat_components_cellwise(A, sycl_target, cell_count_t, 1,
                                         Kernel::maximum<REAL>(),
                                         Kernel::maximum<INT>());

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, reduce_dat_components_cellwise_sub_group) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  A->add_particle_dat(Sym<INT>("FOO"), 4);
  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  wrapper_reduce_dat_components_cellwise(aa, sycl_target, cell_count_t, 2,
                                         Kernel::plus<REAL>(),
                                         Kernel::plus<INT>());
  wrapper_reduce_dat_components_cellwise(aa, sycl_target, cell_count_t, 2,
                                         Kernel::minimum<REAL>(),
                                         Kernel::minimum<INT>());
  wrapper_reduce_dat_components_cellwise(aa, sycl_target, cell_count_t, 2,
                                         Kernel::maximum<REAL>(),
                                         Kernel::maximum<INT>());

  auto AA = particle_sub_group(A);
  wrapper_reduce_dat_components_cellwise(AA, sycl_target, cell_count_t, 1,
                                         Kernel::plus<REAL>(),
                                         Kernel::plus<INT>());
  wrapper_reduce_dat_components_cellwise(AA, sycl_target, cell_count_t, 1,
                                         Kernel::minimum<REAL>(),
                                         Kernel::minimum<INT>());
  wrapper_reduce_dat_components_cellwise(AA, sycl_target, cell_count_t, 1,
                                         Kernel::maximum<REAL>(),
                                         Kernel::maximum<INT>());

  sycl_target->free();
  A->domain->mesh->free();
}
