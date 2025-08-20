#include "include/test_neso_particles.hpp"

TEST(DSMC, cellwise_pair_list) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int cell_count = 19;
  auto cellwise_pair_list =
      std::make_shared<DSMC::CellwisePairList>(sycl_target, cell_count);

  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  std::mt19937 rng(522342);
  std::uniform_int_distribution<int> dist(0, cell_count - 1);

  const int num_samples = 100;

  std::map<int, std::pair<std::vector<int>, std::vector<int>>> h_correct;
  std::vector<int> h_c(num_samples);
  std::vector<int> h_i(num_samples);
  std::vector<int> h_j(num_samples);

  for (int ix = 0; ix < num_samples; ix++) {
    h_c[ix] = dist(rng);
    h_i[ix] = dist(rng);
    h_j[ix] = dist(rng);
    h_correct[h_c[ix]].first.push_back(h_i[ix]);
    h_correct[h_c[ix]].second.push_back(h_j[ix]);
  }

  cellwise_pair_list->push_back(h_c, h_i, h_j);

  {
    auto h_to_test = cellwise_pair_list->host_get();

    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_correct[cx], h_to_test[cx]);
    }
  }

  for (int ix = 0; ix < num_samples; ix++) {
    h_c[ix] = dist(rng);
    h_i[ix] = dist(rng);
    h_j[ix] = dist(rng);
    h_correct[h_c[ix]].first.push_back(h_i[ix]);
    h_correct[h_c[ix]].second.push_back(h_j[ix]);
  }

  cellwise_pair_list->push_back(h_c, h_i, h_j);

  {
    auto h_to_test = cellwise_pair_list->host_get();

    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_correct[cx], h_to_test[cx]);
    }
  }

  cellwise_pair_list->clear();
  {
    auto h_list = cellwise_pair_list->host_get();
    ASSERT_EQ(h_list.size(), cell_count);
    for (int cx = 0; cx < cell_count; cx++) {
      ASSERT_EQ(h_list.at(cx).first.size(), 0);
      ASSERT_EQ(h_list.at(cx).second.size(), 0);
    }
  }

  sycl_target->free();
}
