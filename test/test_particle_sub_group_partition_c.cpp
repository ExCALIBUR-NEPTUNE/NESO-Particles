#include "include/test_neso_particles.hpp"

TEST(ParticleGroup, partition_particle_group_invalidation) {
  auto [A_t, sycl_target_t, cell_count_t] = particle_loop_common_2d(3, 16, 32);

  auto A = A_t;
  A->add_particle_dat(Sym<INT>("PARTITION"), 1);
  auto sycl_target = sycl_target_t;
  const int num_partitions = 7;

  auto loop_reset = particle_loop(
      A,
      [=](auto INDEX, auto PARTITION) {
        PARTITION.at(0) =
            (INDEX.cell + INDEX.layer) % static_cast<INT>(num_partitions);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("PARTITION")));

  loop_reset->execute();
  {
    auto sub_groups =
        particle_group_partition(A, Sym<INT>("PARTITION"), num_partitions);
    auto lambda_test = [&]() {
      for (int px = 0; px < num_partitions; px++) {
        ASSERT_TRUE(sub_groups.at(px)->create_if_required());
        ASSERT_FALSE(sub_groups.at(px)->create_if_required());
      }
    };

    lambda_test();
    loop_reset->execute();
    lambda_test();

    A->invalidate_group_version();
    lambda_test();

    auto selector = std::dynamic_pointer_cast<
        ParticleSubGroupImplementation::ParticleGroupPartitionSelector>(
        sub_groups.at(0)->selector);
    auto partitioner = selector->particle_group_partitioner;

    loop_reset->execute();
    ASSERT_TRUE(partitioner->get(nullptr));
    ASSERT_FALSE(partitioner->get(nullptr));

    A->invalidate_group_version();
    ASSERT_TRUE(partitioner->get(nullptr));
    ASSERT_FALSE(partitioner->get(nullptr));
  }

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 3 == 0; },
      Access::read(Sym<INT>("ID")));

  {
    auto sub_groups =
        particle_group_partition(aa, Sym<INT>("PARTITION"), num_partitions);

    auto lambda_test = [&]() {
      for (int px = 0; px < num_partitions; px++) {
        ASSERT_TRUE(sub_groups.at(px)->create_if_required());
        ASSERT_FALSE(sub_groups.at(px)->create_if_required());
      }
    };

    auto loop_reset_aa = particle_loop(
        aa, [=]([[maybe_unused]] auto ID) {}, Access::write(Sym<INT>("ID")));

    lambda_test();
    loop_reset->execute();
    lambda_test();

    loop_reset_aa->execute();
    lambda_test();

    loop_reset_aa->execute();
    ASSERT_TRUE(aa->create_if_required());
    ASSERT_FALSE(aa->create_if_required());

    A->invalidate_group_version();
    lambda_test();

    A->invalidate_group_version();
    ASSERT_TRUE(aa->create_if_required());
    ASSERT_FALSE(aa->create_if_required());

    auto selector = std::dynamic_pointer_cast<
        ParticleSubGroupImplementation::ParticleGroupPartitionSelector>(
        sub_groups.at(0)->selector);
    auto partitioner = selector->particle_group_partitioner;

    loop_reset_aa->execute();
    ASSERT_TRUE(partitioner->get(nullptr));
    ASSERT_FALSE(partitioner->get(nullptr));

    A->invalidate_group_version();
    ASSERT_TRUE(partitioner->get(nullptr));
    ASSERT_FALSE(partitioner->get(nullptr));
  }

  {
    A->clear();
    auto sub_groups =
        particle_group_partition(A, Sym<INT>("PARTITION"), num_partitions);
    auto selector = std::dynamic_pointer_cast<
        ParticleSubGroupImplementation::ParticleGroupPartitionSelector>(
        sub_groups.at(0)->selector);
    auto partitioner = selector->particle_group_partitioner;
    partitioner->get(nullptr);
    for (int px = 0; px < num_partitions; px++) {
      ASSERT_TRUE(sub_groups.at(px)->create_if_required());
      ASSERT_FALSE(sub_groups.at(px)->create_if_required());
    }
  }

  sycl_target->free();
}
