#include "include/test_particle_sub_group.hpp"

namespace {
template <typename KERNEL, typename... ARGS>
inline auto create_atomic_loop(ParticleGroupSharedPtr particle_group,
                               std::shared_ptr<LocalArray<int *>> map_ptrs,
                               KERNEL kernel, ARGS... args

) {

  std::size_t offset = get_env_size_t("TEST_OFFSET", 32);

  return particle_loop(
      "sub_group_selector_0", particle_group,
      [=](auto loop_index, auto k_map_ptrs, auto... user_args) {
        const bool required = kernel(user_args...);
        const INT particle_linear_index = loop_index.get_local_linear_index();
        if (required) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              element_atomic(k_map_ptrs.at(1)[loop_index.cell*offset]);
          const int layer = element_atomic.fetch_add(1);
          k_map_ptrs.at(0)[particle_linear_index] = layer;
        } else {
          k_map_ptrs.at(0)[particle_linear_index] = -1;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(map_ptrs), args...);
}

} // namespace

TEST(ParticleSubGroup, ordering_loop_particle_group) {
  // auto A = subgroup_test_common(4001);
  // auto A = subgroup_test_common(500000);
  // auto A = subgroup_test_common(1000001);
  auto A = subgroup_test_common(512 * 100);
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  auto d_counts =
      std::make_shared<BufferDevice<int>>(sycl_target, cell_count, 0);
  auto d_layers =
      std::make_shared<BufferDevice<int>>(sycl_target, A->get_npart_local(), 0);

  int *hd_cell_counts = d_counts->ptr;
  int *hd_layers = d_layers->ptr;

  auto loop_scan = Private::particle_loop_selector_ordering(
      &hd_cell_counts, &hd_layers, A,
      [=](auto ID) -> bool { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  loop_scan->execute();

  auto aa = particle_sub_group(
      A, [=](auto ID) -> bool { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  auto h_counts = std::vector<int>(cell_count);
  sycl_target->queue
      .memcpy(h_counts.data(), d_counts->ptr, sizeof(int) * cell_count)
      .wait_and_throw();

  auto test_cell_dat =
      std::make_shared<CellDatConst<int>>(sycl_target, cell_count, 1, 1);
  test_cell_dat->fill(0);

  particle_loop(
      aa, [=](auto CDC) { CDC.combine(0, 0, 1); },
      Access::reduce(test_cell_dat, Kernel::plus<int>()))
      ->execute();

  auto h_test_cell_dat = test_cell_dat->get_all_cells();

  for (int cx = 0; cx < cell_count; cx++) {
    int count = 0;
    auto ID = A->get_cell(Sym<INT>("ID"), cx);
    const int nrow = ID->nrow;
    for (int rx = 0; rx < nrow; rx++) {
      if (ID->at(rx, 0) % 2 == 0) {
        count++;
      }
    }

    ASSERT_EQ(h_counts.at(cx), aa->get_npart_cell(cx));
    ASSERT_EQ(h_counts.at(cx), h_test_cell_dat[cx]->at(0, 0));
    ASSERT_EQ(h_counts.at(cx), count);
  }

  std::shared_ptr<LocalArray<int *>> map_ptrs =
      std::make_shared<LocalArray<int *>>(sycl_target, 2);

  map_ptrs->set({d_layers->ptr, d_counts->ptr});

  auto loop_atomic = create_atomic_loop(
      A, map_ptrs, [=](auto ID) -> bool { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  loop_atomic->execute();

  int Ntest = 100;

  auto t0 = profile_timestamp();
  for (int ix = 0; ix < Ntest; ix++) {
    loop_atomic->execute();
  }
  nprint("atomic:", profile_elapsed(t0, profile_timestamp()));

  t0 = profile_timestamp();
  for (int ix = 0; ix < Ntest; ix++) {
    loop_scan->execute();
  }
  nprint("scan  :", profile_elapsed(t0, profile_timestamp()));

  A->free();
  sycl_target->free();
  mesh->free();
}

inline void bench(const int N){

  auto A = subgroup_test_common(512 * N);
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;
  const int cell_count = mesh->get_cell_count();

  std::size_t offset = get_env_size_t("TEST_OFFSET", 32);

  auto d_counts =
      std::make_shared<BufferDevice<int>>(sycl_target, cell_count * offset + cell_count, 0);
  auto d_layers =
      std::make_shared<BufferDevice<int>>(sycl_target, A->get_npart_local(), 0);

  int *hd_cell_counts = d_counts->ptr;
  int *hd_layers = d_layers->ptr;

  auto loop_scan = Private::particle_loop_selector_ordering(
      &hd_cell_counts, &hd_layers, A,
      [=](auto ID) -> bool { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  loop_scan->execute();

  std::shared_ptr<LocalArray<int *>> map_ptrs =
      std::make_shared<LocalArray<int *>>(sycl_target, 2);

  map_ptrs->set({d_layers->ptr, d_counts->ptr});

  auto loop_atomic = create_atomic_loop(
      A, map_ptrs, [=](auto ID) -> bool { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  loop_atomic->execute();


  auto k_counts = d_counts->ptr;
  auto lambda_pre = [&](){
    sycl_target->queue.parallel_for(
        sycl::range<1>(cell_count),
        [=](auto idx){
          k_counts[cell_count + idx * offset] = k_counts[idx];
        }
    ).wait_and_throw();
  };
  auto lambda_post = [&](){
    sycl_target->queue.parallel_for(
        sycl::range<1>(cell_count),
        [=](auto idx){
          k_counts[idx] = k_counts[cell_count + idx * offset];
        }
    ).wait_and_throw();
  };

  int Ntest = 100;

  nprint("------------------------------------------");
  nprint(A->get_npart_local());
  auto t0 = profile_timestamp();
  for (int ix = 0; ix < Ntest; ix++) {
    lambda_pre();
    loop_atomic->execute();
    lambda_post();
  }
  nprint("N:", N, "atomic:", profile_elapsed(t0, profile_timestamp()));

  t0 = profile_timestamp();
  for (int ix = 0; ix < Ntest; ix++) {
    loop_scan->execute();
  }
  nprint("N:", N, "scan  :", profile_elapsed(t0, profile_timestamp()));

  A->free();
  sycl_target->free();
  mesh->free();
}

TEST(Perf, perf){
  bench(10);
  bench(100);
  bench(200);
  bench(400);
  bench(800);
  bench(1600);
  bench(3200);
}






