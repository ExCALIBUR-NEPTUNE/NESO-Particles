#include "include/test_neso_particles.hpp"

namespace {
template <typename GROUP_TYPE>
void wrapper_reduce_dat_component_cellwise(std::shared_ptr<GROUP_TYPE> A,
                                           SYCLTargetSharedPtr sycl_target,
                                           const int cell_count_t,
                                           const int mask) {
  auto Vto_test =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  auto Vcorrect =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  Vto_test->fill(0.0);
  Vcorrect->fill(0.0);

  auto particle_group = get_particle_group(A);

  particle_loop(
      A,
      [=](auto V, auto VCDC) {
        for (int dx = 0; dx < 3; dx++) {
          VCDC.fetch_add(dx, 0, V.at(dx));
        }
      },
      Access::read(Sym<REAL>("V")), Access::add(Vcorrect))
      ->execute();

  for (int dx = 0; dx < 3; dx++) {
    reduce_dat_component_cellwise(A, Sym<REAL>("V"), dx, Vto_test, dx, 0,
                                  Kernel::plus<REAL>{});
  }

  {
    auto correct = Vcorrect->get_all_cells();
    auto to_test = Vto_test->get_all_cells();

    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      auto V = particle_group->get_cell(Sym<REAL>("V"), cellx);
      auto ID = particle_group->get_cell(Sym<INT>("ID"), cellx);
      const auto nrow = V->nrow;
      for (int dx = 0; dx < 3; dx++) {
        REAL value = 0.0;

        for (int rx = 0; rx < nrow; rx++) {
          if (ID->at(rx, 0) % mask == 0) {
            value += V->at(rx, dx);
          }
        }

        ASSERT_TRUE(relative_error(correct.at(cellx)->at(dx, 0), value) <
                    1.0e-10);
        ASSERT_TRUE(relative_error(correct.at(cellx)->at(dx, 0),
                                   to_test.at(cellx)->at(dx, 0)) < 1.0e-10);
      }
    }
  }

  auto IDto_test =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 1, 1);
  auto IDcorrect =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 1, 1);
  Vto_test->fill(0);
  Vcorrect->fill(0);

  particle_loop(
      A, [=](auto ID, auto IDCDC) { IDCDC.fetch_add(0, 0, ID.at(0)); },
      Access::read(Sym<INT>("ID")), Access::add(IDcorrect))
      ->execute();

  reduce_dat_component_cellwise(A, Sym<INT>("ID"), 0, IDto_test, 0, 0,
                                Kernel::plus<INT>{});

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

  wrapper_reduce_dat_component_cellwise(A, sycl_target, cell_count_t, 1);

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, reduce_dat_component_cellwise_sub_group) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  wrapper_reduce_dat_component_cellwise(aa, sycl_target, cell_count_t, 2);
  wrapper_reduce_dat_component_cellwise(particle_sub_group(A), sycl_target,
                                        cell_count_t, 1);

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, reduce_dat_components_cellwise) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto Vto_test =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  auto Vcorrect =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count_t, 3, 1);
  Vto_test->fill(0.0);
  Vcorrect->fill(0.0);

  particle_loop(
      A,
      [=](auto V, auto VCDC) {
        for (int dx = 0; dx < 3; dx++) {
          VCDC.fetch_add(dx, 0, V.at(dx));
        }
      },
      Access::read(Sym<REAL>("V")), Access::add(Vcorrect))
      ->execute();

  reduce_dat_components_cellwise(A, Sym<REAL>("V"), Vto_test,
                                 Kernel::plus<REAL>{});

  {
    auto correct = Vcorrect->get_all_cells();
    auto to_test = Vto_test->get_all_cells();

    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      auto V = A->get_cell(Sym<REAL>("V"), cellx);
      for (int dx = 0; dx < 3; dx++) {

        REAL value = 0.0;
        const auto nrow = V->nrow;
        for (int rx = 0; rx < nrow; rx++) {
          value += V->at(rx, dx);
        }

        ASSERT_TRUE(relative_error(correct.at(cellx)->at(dx, 0), value) <
                    1.0e-10);
        ASSERT_TRUE(relative_error(correct.at(cellx)->at(dx, 0),
                                   to_test.at(cellx)->at(dx, 0)) < 1.0e-10);
      }
    }
  }

  A->add_particle_dat(Sym<INT>("FOO"), 4);

  auto FOOto_test =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 2, 2);
  auto FOOcorrect =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count_t, 2, 2);
  Vto_test->fill(0);
  Vcorrect->fill(0);

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
        FOOCDC.fetch_add(0, 0, FOO.at(0));
        FOOCDC.fetch_add(0, 1, FOO.at(1));
        FOOCDC.fetch_add(1, 0, FOO.at(2));
        FOOCDC.fetch_add(1, 1, FOO.at(3));
      },
      Access::read(Sym<INT>("FOO")), Access::add(FOOcorrect))
      ->execute();

  reduce_dat_components_cellwise(A, Sym<INT>("FOO"), FOOto_test,
                                 Kernel::plus<INT>{});

  {
    auto correct = FOOcorrect->get_all_cells();
    auto to_test = FOOto_test->get_all_cells();

    for (int cellx = 0; cellx < cell_count_t; cellx++) {
      auto FOO = A->get_cell(Sym<INT>("FOO"), cellx);

      int index = 0;
      for (int rx = 0; rx < 2; rx++) {
        for (int cx = 0; cx < 2; cx++) {

          INT value = 0;
          const auto nrow = FOO->nrow;
          for (int px = 0; px < nrow; px++) {
            value += FOO->at(px, index);
          }
          ASSERT_EQ(correct.at(cellx)->at(rx, cx), value);
          ASSERT_EQ(correct.at(cellx)->at(rx, cx),
                    to_test.at(cellx)->at(rx, cx));

          index++;
        }
      }
    }
  }

  sycl_target->free();
  A->domain->mesh->free();
}
