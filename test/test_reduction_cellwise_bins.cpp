#include "include/test_neso_particles.hpp"

TEST(ReductionContextCellwiseBins, base) {
  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_common_2d(511, 16, 32);
  auto A = A_t;
  const int cell_count = A->domain->mesh->get_cell_count();

  A->add_particle_dat(Sym<INT>("BIN"), 2);
  const int max_num_bins = 100;
  auto sycl_target = sycl_target_t;

  auto lambda_test = [&](auto g) {
    particle_loop(
        g,
        [=](auto INDEX, auto BIN) { BIN.at(1) = INDEX.layer % max_num_bins; },
        Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("BIN")))
        ->execute();

    auto partition = get_index_map<2, 1>(sycl_target);
    partition_mesh_cells_bins(g, max_num_bins, Sym<INT>("BIN"), 1, partition);
    auto reduction_context =
        std::make_shared<ReductionContextCellwiseBins>(g, partition);

    auto cdc = std::make_shared<CellDatConst<int>>(sycl_target, cell_count,
                                                   max_num_bins, 1);

    // particle loop here

    reduction_context->free();
    restore_index_map(sycl_target, partition);
  };

  lambda_test(A);

  auto aa = particle_sub_group(
      A, [=](auto ID) { return ID.at(0) % 2 == 0; },
      Access::read(Sym<INT>("ID")));
  lambda_test(aa);

  sycl_target_t->free();
  A->domain->mesh->free();
}
