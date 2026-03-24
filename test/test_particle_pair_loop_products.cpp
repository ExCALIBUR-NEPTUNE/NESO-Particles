#include "include/test_neso_particles.hpp"

namespace {

struct TestDescendantProducts : public DescendantProducts {

  template <typename... ARGS>
  TestDescendantProducts(ARGS... args) : DescendantProducts(args...) {}

  MAKE_GETTER_METHOD(d_parent_cells)
  MAKE_GETTER_METHOD(d_parent_layers)
};
} // namespace

TEST(ParticlePairLoopBlock, descendant_products) {

  const int num_products = 2;
  const int npart_cell = 257;
  const int ndim = 2;
  const int nx = 16;
  const int ny = 33;
  const int nz = 48;

  auto [A_t, sycl_target_t, cell_count_t] =
      particle_loop_create_common(npart_cell, ndim, nx, ny, nz);
  auto A = A_t;
  auto sycl_target = sycl_target_t;
  auto cell_count = cell_count_t;

  auto product_spec = product_matrix_spec(A->get_particle_spec());
  auto dp_a = std::make_shared<TestDescendantProducts>(
      sycl_target, product_spec, num_products);
  auto dp_b = std::make_shared<TestDescendantProducts>(
      sycl_target, product_spec, num_products);

  const int num_samples = cell_count * npart_cell * 0.2;
  std::vector<int> h_c(num_samples);
  std::vector<int> h_i(num_samples);
  std::vector<int> h_j(num_samples);

  std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
  std::uniform_int_distribution<int> dist_cell(0, cell_count - 1);

  for (int ix = 0; ix < num_samples; ix++) {
    const int cell = dist_cell(rng);
    h_c[ix] = cell;
    const int npart_cell = A->get_npart_cell(cell);
    std::uniform_int_distribution<int> dist_layer(0, npart_cell - 1);
    h_i[ix] = dist_layer(rng);
    h_j[ix] = dist_layer(rng);
  }

  auto cellwise_pair_list =
      std::make_shared<CellwisePairListSimple>(sycl_target, cell_count);
  cellwise_pair_list->push_back(h_c, h_i, h_j);

  const int num_pairs = cellwise_pair_list->get_num_pairs();
  dp_a->reset(num_pairs);
  dp_b->reset(num_pairs);

  ASSERT_EQ(num_pairs, dp_a->num_parent_particles);
  ASSERT_EQ(num_pairs, dp_b->num_parent_particles);
  ASSERT_EQ(num_products, dp_a->num_products_per_parent);
  ASSERT_EQ(num_products, dp_b->num_products_per_parent);

  particle_pair_loop(
      "particle_pair_loop_test",
      {CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
          A, A, cellwise_pair_list)},
      [=](auto INDEX_a, auto INDEX_b, auto DP_a, auto DP_b) {
        for (int child = 0; child < num_products; child++) {
          DP_a.set_parent(INDEX_a, child);
          DP_b.set_parent(INDEX_b, child);

          DP_a.at_real(INDEX_a, child, 0, 0) =
              INDEX_a.get_loop_linear_index() + child;
          DP_b.at_real(INDEX_b, child, 0, 0) =
              INDEX_b.get_loop_linear_index() + child;

          DP_a.at_int(INDEX_a, child, 0, 0) =
              INDEX_a.get_loop_linear_index() + child;
          DP_b.at_int(INDEX_b, child, 0, 0) =
              INDEX_b.get_loop_linear_index() + child;
        }
      },
      Access::A(Access::read(ParticlePairLoopIndex{})),
      Access::B(Access::read(ParticlePairLoopIndex{})),
      Access::A(
          Access::write(std::dynamic_pointer_cast<DescendantProducts>(dp_a))),
      Access::B(
          Access::write(std::dynamic_pointer_cast<DescendantProducts>(dp_b))))
      ->execute();

  auto h_dp_a = dp_a->get();
  auto h_dp_b = dp_b->get();

  Sym<REAL> sym_real = product_spec->syms_real.at(0);
  Sym<INT> sym_int = product_spec->syms_int.at(0);

  auto &d_parent_cells_a = dp_a->get_d_parent_cells();
  auto &d_parent_cells_b = dp_b->get_d_parent_cells();
  auto &d_parent_layers_a = dp_a->get_d_parent_layers();
  auto &d_parent_layers_b = dp_b->get_d_parent_layers();

  std::vector<INT> h_parent_cells_a(num_pairs * num_products);
  std::vector<INT> h_parent_cells_b(num_pairs * num_products);
  std::vector<INT> h_parent_layers_a(num_pairs * num_products);
  std::vector<INT> h_parent_layers_b(num_pairs * num_products);

  const std::size_t num_bytes = num_pairs * num_products * sizeof(INT);
  sycl_target->queue
      .memcpy(h_parent_cells_a.data(), d_parent_cells_a->ptr, num_bytes)
      .wait_and_throw();
  sycl_target->queue
      .memcpy(h_parent_cells_b.data(), d_parent_cells_b->ptr, num_bytes)
      .wait_and_throw();
  sycl_target->queue
      .memcpy(h_parent_layers_a.data(), d_parent_layers_a->ptr, num_bytes)
      .wait_and_throw();
  sycl_target->queue
      .memcpy(h_parent_layers_b.data(), d_parent_layers_b->ptr, num_bytes)
      .wait_and_throw();

  auto h_pair_list = cellwise_pair_list->get_host_pair_list();

  int pair_index = 0;
  for (auto &cell_pair_list : h_pair_list) {
    const auto cell = cell_pair_list.first;
    for (auto &wave_pairs : cell_pair_list.second) {
      const auto &pairs = wave_pairs.second;
      const int num_pairs_wave = pairs.first.size();
      for (int pairx = 0; pairx < num_pairs_wave; pairx++) {
        const int a = pairs.first.at(pairx);
        const int b = pairs.second.at(pairx);

        for (int child = 0; child < num_products; child++) {
          const int child_linear_index = child * num_pairs + pair_index;

          ASSERT_EQ(h_parent_cells_a.at(child_linear_index), cell);
          ASSERT_EQ(h_parent_cells_b.at(child_linear_index), cell);
          ASSERT_EQ(h_parent_layers_a.at(child_linear_index), a);
          ASSERT_EQ(h_parent_layers_b.at(child_linear_index), b);

          ASSERT_EQ(h_dp_a->at(sym_real, child_linear_index, 0),
                    pair_index + child);
          ASSERT_EQ(h_dp_b->at(sym_real, child_linear_index, 0),
                    pair_index + child);
          ASSERT_EQ(h_dp_a->at(sym_int, child_linear_index, 0),
                    pair_index + child);
          ASSERT_EQ(h_dp_b->at(sym_int, child_linear_index, 0),
                    pair_index + child);
        }

        pair_index++;
      }
    }
  }
  ASSERT_EQ(pair_index, num_pairs);

  sycl_target->free();
  A->domain->mesh->free();
}
