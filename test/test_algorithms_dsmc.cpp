#include "include/test_neso_particles.hpp"

TEST(DSMC, all_to_all_looping_host) {

  const int workgroup_size = 4;

  for (int num_particles : {0, 1, 3, 4, 7, 8, 31}) {

    std::vector<int> seen_entries(num_particles * num_particles);
    std::fill(seen_entries.begin(), seen_entries.end(), 0);
    std::vector<int> num_sync_calls(workgroup_size);
    std::fill(num_sync_calls.begin(), num_sync_calls.end(), 0);

    for (int workgroup_item = 0; workgroup_item < workgroup_size;
         workgroup_item++) {
      auto loop_context =
          DSMC::CellwiseAllToAll(workgroup_size, workgroup_item);

      loop_context.apply(
          num_particles,
          [&](auto i, auto j) {
            seen_entries.at(i + num_particles * j)++;
            seen_entries.at(j + num_particles * i)++;
          },
          [&]() { num_sync_calls.at(workgroup_item)++; });
    }

    for (int ix = 0; ix < workgroup_size; ix++) {
      ASSERT_EQ(num_sync_calls.at(ix), num_sync_calls.at(0));
    }

    for (int rowx = 0; rowx < num_particles; rowx++) {
      for (int colx = 0; colx < num_particles; colx++) {
        ASSERT_EQ(seen_entries.at(rowx + num_particles * colx),
                  rowx == colx ? 0 : 1);
      }
    }
  }
}

TEST(DSMC, all_to_all_looping_device) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

#ifdef __ACPP__
  const int cpu_workgroup_size = 1;
  const bool use_barrier = sycl_target->device.is_cpu() ? false : true;
#else
  const int cpu_workgroup_size = 32;
  constexpr bool use_barrier = true;
#endif

  const int workgroup_size =
      sycl_target->device.is_cpu() ? cpu_workgroup_size : 256;

  for (int num_particles : {0, 1, 3, 4, 7, 8, 31, 255, 1023, 1024}) {

    std::vector<int> seen_entries(num_particles * num_particles);
    std::fill(seen_entries.begin(), seen_entries.end(), 0);
    BufferDevice<int> d_seen_entries(sycl_target, seen_entries);
    auto k_seen_entries = d_seen_entries.ptr;

    sycl_target->queue
        .parallel_for(sycl::nd_range<1>(sycl::range<1>(workgroup_size),
                                        sycl::range<1>(workgroup_size)),
                      [=](sycl::nd_item<1> idx) {
                        DSMC::CellwiseAllToAll loop_context(
                            idx.get_local_range(0), idx.get_local_id(0));

                        loop_context.apply(
                            num_particles,
                            [&](auto i, auto j) {
                              k_seen_entries[i + num_particles * j]++;
                              k_seen_entries[j + num_particles * i]++;
                            },
                            [=]() {
                              if (use_barrier) {
                                sycl::group_barrier(idx.get_group());
                              }
                            });
                      })
        .wait_and_throw();

    seen_entries = d_seen_entries.get();

    for (int rowx = 0; rowx < num_particles; rowx++) {
      for (int colx = 0; colx < num_particles; colx++) {
        ASSERT_EQ(seen_entries.at(rowx + num_particles * colx),
                  rowx == colx ? 0 : 1);
      }
    }
  }

  sycl_target->free();
}
