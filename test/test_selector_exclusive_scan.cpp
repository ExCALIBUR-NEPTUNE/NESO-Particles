#include "include/test_particle_sub_group.hpp"

TEST(SubGroupSelectorExclusiveScan, get_npart_cell_device) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();
  const int cell_count = mesh->get_cell_count();

  int *d_a, *d_b;
  INT *d_A, *d_B;
  auto lambda_test = [&]() {
    sycl_target->queue
        .parallel_for(sycl::range<1>(cell_count),
                      [=](auto ix) {
                        NESO_KERNEL_ASSERT(d_a[ix] == d_b[ix], k_ep);
                        NESO_KERNEL_ASSERT(d_A[ix] == d_B[ix], k_ep);
                      })
        .wait_and_throw();
    ASSERT_FALSE(ep.get_flag());
  };

  auto AA = particle_sub_group(A);
  d_a = Private::get_npart_cell_device_ptr(A);
  d_b = Private::get_npart_cell_device_ptr(AA);
  d_A = Private::get_npart_cell_es_device_ptr(A);
  d_B = Private::get_npart_cell_es_device_ptr(AA);
  lambda_test();

  auto aa = static_particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 3 == 0; },
      Access::read(Sym<INT>("ID")));
  auto bb = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 3 == 0; },
      Access::read(Sym<INT>("ID")));

  d_a = Private::get_npart_cell_device_ptr(aa);
  d_b = Private::get_npart_cell_device_ptr(bb);
  d_A = Private::get_npart_cell_es_device_ptr(aa);
  d_B = Private::get_npart_cell_es_device_ptr(bb);
  lambda_test();

  auto cc = particle_sub_group(A, [=]() { return true; });

  d_a = Private::get_npart_cell_device_ptr(A);
  d_b = Private::get_npart_cell_device_ptr(cc);
  d_A = Private::get_npart_cell_es_device_ptr(A);
  d_B = Private::get_npart_cell_es_device_ptr(cc);
  lambda_test();

  sycl_target->free();
  mesh->free();
}

TEST(SubGroupSelectorExclusiveScan, selector) {
  auto A = subgroup_test_common();
  auto domain = A->domain;
  auto mesh = domain->mesh;
  auto sycl_target = A->sycl_target;

  auto selector_exscan = std::make_shared<
      ParticleSubGroupImplementation::SubGroupSelectorExclusiveScan>(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  auto selector_atomic =
      std::make_shared<ParticleSubGroupImplementation::SubGroupSelector>(
          A, [=](auto ID) { return ID.at(0) % 2 == 0; },
          Access::read(Sym<INT>("ID")));

  ParticleSubGroupImplementation::Selection selection_exscan;
  ParticleSubGroupImplementation::Selection selection_atomic;

  selector_exscan->create(&selection_exscan);
  selector_atomic->create(&selection_atomic);

  auto host_exscan =
      get_host_map_cells_to_particles(sycl_target, selection_exscan);
  auto host_atomic =
      get_host_map_cells_to_particles(sycl_target, selection_atomic);

  const auto cell_count = mesh->get_cell_count();

  ASSERT_EQ(selection_exscan.npart_local, selection_atomic.npart_local);
  ASSERT_EQ(selection_exscan.ncell, selection_atomic.ncell);

  std::vector<int> h_npart_cell_exscan(cell_count);
  std::vector<int> h_npart_cell_atomic(cell_count);

  sycl_target->queue
      .memcpy(h_npart_cell_exscan.data(), selection_exscan.d_npart_cell,
              cell_count * sizeof(int))
      .wait_and_throw();
  sycl_target->queue
      .memcpy(h_npart_cell_atomic.data(), selection_atomic.d_npart_cell,
              cell_count * sizeof(int))
      .wait_and_throw();

  std::vector<INT> h_npart_cell_es_exscan(cell_count);
  std::vector<INT> h_npart_cell_es_atomic(cell_count);

  sycl_target->queue
      .memcpy(h_npart_cell_es_exscan.data(), selection_exscan.d_npart_cell_es,
              cell_count * sizeof(int))
      .wait_and_throw();
  sycl_target->queue
      .memcpy(h_npart_cell_es_atomic.data(), selection_atomic.d_npart_cell_es,
              cell_count * sizeof(int))
      .wait_and_throw();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    std::set<INT> layers_exscan;
    std::set<INT> layers_atomic;
    for (auto ax : host_atomic[cellx]) {
      ASSERT_EQ(layers_atomic.count(ax), 0);
      layers_atomic.insert(ax);
    }
    for (auto ax : host_exscan[cellx]) {
      ASSERT_EQ(layers_exscan.count(ax), 0);
      layers_exscan.insert(ax);
    }

    ASSERT_EQ(layers_exscan, layers_atomic);
    ASSERT_EQ(selection_exscan.h_npart_cell[cellx],
              selection_atomic.h_npart_cell[cellx]);
    ASSERT_EQ(h_npart_cell_exscan[cellx], h_npart_cell_atomic[cellx]);
    ASSERT_EQ(h_npart_cell_es_exscan[cellx], h_npart_cell_es_atomic[cellx]);
  }

  sycl_target->free();
  mesh->free();
}
