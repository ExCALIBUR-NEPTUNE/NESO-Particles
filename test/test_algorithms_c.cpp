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

TEST(Algorithms, cellwise_broadcast) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  const int cell_count = cell_count_t;

  A->add_particle_dat(Sym<INT>("FOO"), 4);
  A->add_particle_dat(Sym<REAL>("BAR"), 3);

  auto lambda_test = [&](auto AA, auto sym, auto component, auto values,
                         auto mask) {
    cellwise_broadcast(AA, sym, component, values);
    auto B = get_particle_group(AA);
    for (int cx = 0; cx < cell_count; cx++) {
      auto V = B->get_cell(sym, cx);
      auto ID = B->get_cell(Sym<INT>("ID"), cx);
      const int nrow = V->nrow;
      for (int rx = 0; rx < nrow; rx++) {
        if (mask && (ID->at(rx, 0) % 2)) {
          ASSERT_EQ(V->at(rx, component), values.at(cx));
        } else {
          ASSERT_EQ(V->at(rx, component), values.at(cx));
        }
      }
    }
  };

  std::vector<INT> values_INT(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    values_INT[cx] = cx + 1;
  }
  std::vector<REAL> values_REAL(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    values_REAL[cx] = cx + 0.123;
  }
  std::vector<int> values_int(cell_count);
  for (int cx = 0; cx < cell_count; cx++) {
    values_int[cx] = cx + 1;
  }

  lambda_test(A, Sym<INT>("FOO"), 1, values_INT, false);
  lambda_test(A, Sym<INT>("FOO"), 0, values_int, false);
  lambda_test(A, Sym<REAL>("BAR"), 1, values_REAL, false);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2; }, Access::read(Sym<INT>("ID")));

  lambda_test(aa, Sym<INT>("FOO"), 1, values_INT, true);
  lambda_test(aa, Sym<INT>("FOO"), 0, values_int, true);
  lambda_test(aa, Sym<REAL>("BAR"), 1, values_REAL, true);

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, get_npart_cell) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);
  const int cell_count = cell_count_t;

  auto cdc_a =
      std::make_shared<CellDatConst<int>>(sycl_target, cell_count, 1, 1);
  get_npart_cell(A, cdc_a);

  auto h_cdc_a = cdc_a->get_all_cells();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(h_cdc_a.at(cellx)->at(0, 0), A->get_npart_cell(cellx));
  }

  auto cdc_b =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, 2, 3);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2; }, Access::read(Sym<INT>("ID")));

  get_npart_cell(aa, cdc_b, 1, 2);

  auto h_cdc_b = cdc_b->get_all_cells();
  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_EQ(h_cdc_b.at(cellx)->at(1, 2), aa->get_npart_cell(cellx));
  }

  sycl_target->free();
  A->domain->mesh->free();
}

TEST(Algorithms, cell_dat_const_loop_element_wise_base) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int cell_count = 61;
  int nrow = 3;
  int ncol = 7;

  auto a =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);
  auto b =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);
  auto c =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);
  auto d =
      std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);

  std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
  std::uniform_real_distribution<REAL> dist(1.0, 4.0);
  auto h_a = a->get_all_cells();
  auto h_b = b->get_all_cells();
  auto h_c = c->get_all_cells();
  auto h_d = d->get_all_cells();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int colx = 0; colx < ncol; colx++) {
      for (int rowx = 0; rowx < nrow; rowx++) {
        const auto ta = dist(rng);
        const auto tb = dist(rng);
        const auto tc = dist(rng);
        h_a.at(cellx)->at(rowx, colx) = ta;
        h_b.at(cellx)->at(rowx, colx) = tb;
        h_c.at(cellx)->at(rowx, colx) = tc;
        h_d.at(cellx)->at(rowx, colx) = ta * tb + tc;
      }
    }
  }

  a->set_all_cells(h_a);
  b->set_all_cells(h_b);
  c->set_all_cells(h_c);
  d->fill(0);
  cell_dat_const_loop_element_wise(
      d, [=](auto a, auto b, auto c) { return a * b + c; }, a, b, c);

  auto h_d_to_test = d->get_all_cells();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int colx = 0; colx < ncol; colx++) {
      for (int rowx = 0; rowx < nrow; rowx++) {
        const auto correct = h_d.at(cellx)->at(rowx, colx);
        const auto to_test = h_d_to_test.at(cellx)->at(rowx, colx);
        ASSERT_NEAR(correct, to_test, 1.0e-14);
      }
    }
  }

  cell_dat_const_loop_element_wise(d, [=](auto d) { return d * 2; }, d);

  h_d_to_test = d->get_all_cells();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int colx = 0; colx < ncol; colx++) {
      for (int rowx = 0; rowx < nrow; rowx++) {
        const auto correct = 2.0 * h_d.at(cellx)->at(rowx, colx);
        const auto to_test = h_d_to_test.at(cellx)->at(rowx, colx);
        ASSERT_NEAR(correct, to_test, 1.0e-14);
      }
    }
  }

  sycl_target->free();
}
