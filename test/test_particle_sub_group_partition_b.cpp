#include "include/test_neso_particles.hpp"

TEST(ParticleGroup, partition_particle_group_reference_counting) {
  auto [A_t, sycl_target_t, cell_count_t] = particle_loop_common_2d(3, 16, 32);

  auto A = A_t;
  A->add_particle_dat(Sym<INT>("PARTITION"), 1);
  auto sycl_target = sycl_target_t;
  const int num_partitions = 7;

  particle_loop(
      A,
      [=](auto INDEX, auto PARTITION) {
        PARTITION.at(0) =
            (INDEX.cell + INDEX.layer) % static_cast<INT>(num_partitions);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("PARTITION")))
      ->execute();

  auto sub_groups =
      particle_group_partition(A, Sym<INT>("PARTITION"), num_partitions);

  for (int px = 0; px < num_partitions; px++) {
    ASSERT_EQ(sub_groups.at(px).use_count(), 1);
    ASSERT_EQ(sub_groups.at(px)->selector.use_count(), 1);
    auto selector = std::dynamic_pointer_cast<
        ParticleSubGroupImplementation::ParticleGroupPartitionSelector>(
        sub_groups.at(px)->selector);
    ASSERT_EQ(selector.use_count(), 2);
    ASSERT_EQ(selector->particle_group_partitioner.use_count(), num_partitions);
  }

  auto particle_group_partitioner =
      std::dynamic_pointer_cast<
          ParticleSubGroupImplementation::ParticleGroupPartitionSelector>(
          sub_groups.at(0)->selector)
          ->particle_group_partitioner;
  ASSERT_EQ(particle_group_partitioner.use_count(), num_partitions + 1);

  for (int px = 0; px < num_partitions; px++) {
    sub_groups.at(px) = nullptr;
    ASSERT_EQ(particle_group_partitioner.use_count(), num_partitions - px);
  }

  // This last reference is particle_group_partitioner itself
  ASSERT_EQ(particle_group_partitioner.use_count(), 1);

  sycl_target->free();
}

TEST(ParticleGroup, partition_particle_group_deleting) {
  auto [A_t, sycl_target_t, cell_count_t] = particle_loop_common_2d(10, 16, 32);

  auto A = A_t;
  A->add_particle_dat(Sym<INT>("PARTITION"), 1);
  auto sycl_target = sycl_target_t;
  const int num_partitions = 7;

  particle_loop(
      A,
      [=](auto INDEX, auto PARTITION) {
        PARTITION.at(0) =
            (INDEX.cell + INDEX.layer) % static_cast<INT>(num_partitions);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("PARTITION")))
      ->execute();

  auto sub_groups =
      particle_group_partition(A, Sym<INT>("PARTITION"), num_partitions);

  auto ep = std::make_shared<ErrorPropagate>(sycl_target);
  auto k_ep = ep->device_ptr();

  auto particle_group_partitioner =
      std::dynamic_pointer_cast<
          ParticleSubGroupImplementation::ParticleGroupPartitionSelector>(
          sub_groups.at(0)->selector)
          ->particle_group_partitioner;

  for (int px = 0; px < num_partitions; px++) {
    sub_groups.at(px) = nullptr;
    for (int rx = px + 1; rx < num_partitions; rx++) {
      particle_loop(
          sub_groups.at(rx),
          [=](auto PARTITION) {
            NESO_KERNEL_ASSERT(PARTITION.at(0) == rx, k_ep);
          },
          Access::read(Sym<INT>("PARTITION")))
          ->execute();
      ASSERT_FALSE(ep->get_flag());
    }
    particle_loop(
        A,
        [=](auto INDEX, auto PARTITION) {
          PARTITION.at(0) = (INDEX.cell + INDEX.layer + px + 1) %
                            static_cast<INT>(num_partitions);
        },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("PARTITION")))
        ->execute();

    auto h_still_exists = particle_group_partitioner->la_still_exists->get();

    // The call to remake the particle group will never happen for the last
    // partition
    if (px < num_partitions - 1) {
      for (int rx = 0; rx <= px; rx++) {
        ASSERT_EQ(h_still_exists.at(rx), 0);
        ASSERT_TRUE(
            particle_group_partitioner->partition_selectors.at(rx).expired());
        ASSERT_TRUE(particle_group_partitioner->sub_group_particle_maps.at(rx)
                        .expired());
      }
      for (int rx = px + 1; rx < num_partitions; rx++) {
        ASSERT_EQ(h_still_exists.at(rx), 1);
        ASSERT_FALSE(
            particle_group_partitioner->partition_selectors.at(rx).expired());
        ASSERT_FALSE(particle_group_partitioner->sub_group_particle_maps.at(rx)
                         .expired());
      }
    }
  }

  for (int px = 0; px < num_partitions; px++) {
    ASSERT_TRUE(
        particle_group_partitioner->partition_selectors.at(px).expired());
    ASSERT_TRUE(
        particle_group_partitioner->sub_group_particle_maps.at(px).expired());
  }

  sycl_target->free();
}
