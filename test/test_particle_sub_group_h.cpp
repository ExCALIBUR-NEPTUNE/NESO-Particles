#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, range_cell_base) {
  auto A = subgroup_test_common(4001);
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  constexpr int num_comp = 8;
  A->add_particle_dat(ParticleDat(sycl_target,
                                  ParticleProp(Sym<INT>("TEST"), num_comp),
                                  domain->mesh->get_cell_count()));

  auto ep = error_propagate(sycl_target);
  auto k_ep = ep->device_ptr();

  auto loop_reset = particle_loop(
      A,
      [=](auto TEST) {
        for (int cx = 0; cx < num_comp; cx++) {
          TEST.at(cx) = -1;
        }
      },
      Access::write(Sym<INT>("TEST")));

  auto lambda_test_range = [&](auto parent, const int cell_start,
                               const int cell_end) {
    auto aa = particle_sub_group(parent, cell_start, cell_end);

    particle_loop(
        parent,
        [=](auto INDEX, auto TEST) {
          TEST.at(0) = INDEX.get_loop_linear_index();
          TEST.at(1) = INDEX.get_local_linear_index();
        },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("TEST")))
        ->execute();

    std::set<INT> correct_loop_linear_index;
    std::set<INT> to_test_loop_linear_index;
    INT correct_local_linear_index = 0;
    INT correct_counter = 0;
    for (int cx = 0; cx < cell_count; cx++) {
      auto TEST = A->get_cell(Sym<INT>("TEST"), cx);
      auto ID = A->get_cell(Sym<INT>("ID"), cx);
      const int nrow = TEST->nrow;
      for (int rx = 0; rx < nrow; rx++) {
        if (TEST->at(rx, 0) >= 0) {
          ASSERT_EQ(correct_local_linear_index, TEST->at(rx, 1));
          correct_loop_linear_index.insert(correct_counter++);
          to_test_loop_linear_index.insert(TEST->at(rx, 0));
        }
        correct_local_linear_index++;
      }
      ASSERT_EQ(correct_loop_linear_index, to_test_loop_linear_index);
      correct_loop_linear_index.clear();
      to_test_loop_linear_index.clear();
    }
    particle_loop(
        parent, [=](auto TEST) { TEST.at(1) = -1; },
        Access::write(Sym<INT>("TEST")))
        ->execute();

    particle_loop(
        A,
        [=](auto INDEX, auto TEST) {
          if (TEST.at(0) >= 0) {
            TEST.at(1) = INDEX.cell;
            TEST.at(2) = INDEX.layer;
            TEST.at(5) = INDEX.get_local_linear_index();
          }
        },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("TEST")))
        ->execute(cell_start, cell_end);

    particle_loop(
        aa,
        [=](auto INDEX, auto TEST) {
          TEST.at(3) = INDEX.cell;
          TEST.at(4) = INDEX.layer;
          TEST.at(6) = INDEX.get_local_linear_index();
          TEST.at(7) = INDEX.get_loop_linear_index();
        },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("TEST")))
        ->execute();

    particle_loop(
        A,
        [=](auto TEST) {
          // Is the particle in the parent group
          if (TEST.at(0) >= 0) {
            NESO_KERNEL_ASSERT(TEST.at(1) == TEST.at(3), k_ep);
            NESO_KERNEL_ASSERT(TEST.at(2) == TEST.at(4), k_ep);
            NESO_KERNEL_ASSERT(TEST.at(5) == TEST.at(6), k_ep);
          } else {
            for (int cx = 1; cx < num_comp; cx++) {
              NESO_KERNEL_ASSERT(TEST.at(cx) == -1, k_ep);
            }
          }
        },
        Access::read(Sym<INT>("TEST")))
        ->execute();
    ASSERT_FALSE(ep->get_flag());

    correct_counter = 0;
    for (int cx = cell_start; cx < cell_end; cx++) {
      auto TEST = A->get_cell(Sym<INT>("TEST"), cx);
      const int nrow = TEST->nrow;
      for (int rx = 0; rx < nrow; rx++) {
        if (TEST->at(rx, 0) >= 0) {
          correct_loop_linear_index.insert(correct_counter++);
          to_test_loop_linear_index.insert(TEST->at(rx, 7));
        }
      }
      ASSERT_EQ(correct_loop_linear_index, to_test_loop_linear_index);
      correct_loop_linear_index.clear();
      to_test_loop_linear_index.clear();
    }
  };

  auto lambda_run = [&]() {
    loop_reset->execute();
    lambda_test_range(A, 0, cell_count);
    loop_reset->execute();
    lambda_test_range(A, 0, 1);
    loop_reset->execute();
    lambda_test_range(A, cell_count - 1, cell_count);

    if (cell_count > 4) {
      loop_reset->execute();
      lambda_test_range(A, cell_count - 3, cell_count);
      loop_reset->execute();
      lambda_test_range(A, 1, 3);
    }

    auto Aeven = particle_sub_group(
        A, [=](auto ID) { return ID.at(0) % 2 == 0; },
        Access::read(Sym<INT>("ID")));

    loop_reset->execute();
    lambda_test_range(Aeven, 0, cell_count);

    loop_reset->execute();
    lambda_test_range(Aeven, 0, 1);

    loop_reset->execute();
    lambda_test_range(Aeven, cell_count - 1, cell_count);

    if (cell_count > 4) {
      loop_reset->execute();
      lambda_test_range(Aeven, cell_count - 3, cell_count);
      loop_reset->execute();
      lambda_test_range(Aeven, 1, 3);
    }

    // empty set
    loop_reset->execute();
    lambda_test_range(A, 0, 0);

    // empty set
    loop_reset->execute();
    lambda_test_range(Aeven, 0, 0);
  };

  lambda_run();
  A->clear();
  lambda_run();

  A->free();
  sycl_target->free();
  mesh->free();
}
