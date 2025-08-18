#include "include/test_neso_particles.hpp"

TEST(ParticleGroup, partition_particle_group) {
  auto [A_t, sycl_target_t, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto A = A_t;
  auto sycl_target = sycl_target_t;

  const int cell_count = A->domain->mesh->get_cell_count();
  A->add_particle_dat(Sym<INT>("PARTITION"), 1);

  auto lambda_test = [&](auto aa, auto filter, const std::size_t num_partitions,
                         Sym<INT> sym_extra) -> void {
    particle_loop(
        A,
        [=](auto INDEX, auto PARTITION) {
          PARTITION.at(0) =
              (INDEX.cell + INDEX.layer) % static_cast<INT>(num_partitions);
        },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("PARTITION")))
        ->execute();

    auto sub_groups =
        particle_group_partition(aa, Sym<INT>("PARTITION"), num_partitions);

    auto lambda_test_inner = [&]() {
      auto la_count =
          std::make_shared<LocalArray<int>>(sycl_target, 1 + num_partitions);
      la_count->fill(0);
      auto ep = std::make_shared<ErrorPropagate>(sycl_target);
      auto k_ep = ep->device_ptr();

      for (std::size_t px = 0; px < num_partitions; px++) {
        ASSERT_TRUE(
            sub_groups.at(px)->selector->depends_on(Sym<INT>("PARTITION")));
        ASSERT_TRUE(sub_groups.at(px)->selector->depends_on(sym_extra));

        particle_loop(
            sub_groups.at(px),
            [=](auto PARTITION, auto LA_COUNT) {
              NESO_KERNEL_ASSERT(PARTITION.at(0) == static_cast<INT>(px), k_ep);
              LA_COUNT.fetch_add(0, 1);
              LA_COUNT.fetch_add(1 + PARTITION.at(0), 1);
            },
            Access::read(Sym<INT>("PARTITION")), Access::add(la_count))
            ->execute();
        ASSERT_FALSE(ep->get_flag());

        std::set<std::tuple<int, int>> correct_pairs;
        for (int cx = 0; cx < cell_count; cx++) {
          auto PARTITION = A->get_cell(Sym<INT>("PARTITION"), cx);
          auto ID = A->get_cell(Sym<INT>("ID"), cx);
          const int nrow = PARTITION->nrow;
          for (int rx = 0; rx < nrow; rx++) {
            if ((PARTITION->at(rx, 0) == static_cast<INT>(px)) &&
                filter(ID->at(rx, 0))) {
              std::tuple<int, int> key = {cx, rx};
              ASSERT_EQ(correct_pairs.count(key), 0);
              correct_pairs.insert(key);
            }
          }
        }

        ASSERT_TRUE(selection_is_self_consistent(
            sub_groups.at(px)->selection, cell_count,
            la_count->get().at(1 + px), sycl_target, correct_pairs));
      }

      ASSERT_EQ(la_count->get().at(0), aa->get_npart_local());
    };

    lambda_test_inner();
    particle_loop(
        aa,
        [=](auto PARTITION) {
          PARTITION.at(0) = (PARTITION.at(0) + 1) % num_partitions;
        },
        Access::write(Sym<INT>("PARTITION")))
        ->execute();
    lambda_test_inner();
    particle_loop(
        A, [=](auto ID) { ID.at(0) += 1; }, Access::write(Sym<INT>("ID")))
        ->execute();
    lambda_test_inner();
  };

  lambda_test(
      A, [&]([[maybe_unused]] auto ID) { return true; }, 1,
      Sym<INT>("PARTITION"));
  lambda_test(
      A, [&]([[maybe_unused]] auto ID) { return true; }, 3,
      Sym<INT>("PARTITION"));
  lambda_test(
      A, [&]([[maybe_unused]] auto ID) { return true; }, 7,
      Sym<INT>("PARTITION"));

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 3 == 0; },
      Access::read(Sym<INT>("ID")));
  lambda_test(
      aa, [&]([[maybe_unused]] auto ID) { return ID % 3 == 0; }, 1,
      Sym<INT>("ID"));
  lambda_test(
      aa, [&]([[maybe_unused]] auto ID) { return ID % 3 == 0; }, 2,
      Sym<INT>("ID"));
  lambda_test(
      aa, [&]([[maybe_unused]] auto ID) { return ID % 3 == 0; }, 5,
      Sym<INT>("ID"));

  sycl_target->free();
}
