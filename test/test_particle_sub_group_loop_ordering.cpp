#include "include/test_particle_sub_group.hpp"

TEST(ParticleSubGroup, ordering_loop_particle_group) {
  auto A = subgroup_test_common(4001);
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

  Private::particle_loop_selector_ordering(
      &hd_cell_counts, &hd_layers, A,
      [=](auto ID) -> bool { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")))
      ->execute();

  auto aa = particle_sub_group(
      A, [=](auto ID) -> bool { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));

  auto h_counts = std::vector<int>(cell_count);
  sycl_target->queue
      .memcpy(h_counts.data(), d_counts->ptr, sizeof(int) * cell_count)
      .wait_and_throw();

  for (int cx = 0; cx < cell_count; cx++) {
    ASSERT_EQ(h_counts.at(cx), aa->get_npart_cell(cx));
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
