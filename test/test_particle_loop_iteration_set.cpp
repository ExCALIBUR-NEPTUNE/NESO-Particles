#include "include/test_neso_particles.hpp"

using namespace ParticleLoopImplementation;

TEST(ParticleLoopIterationSet, base) {

  const int nbin = 16;
  const int ncell = 1261;
  const int npart_cell_max = 89;
  const int floor_occupancy = 48;
  const int local_size = 32;

  std::vector<int> h_npart_cell(ncell);

  std::mt19937 rng(522342);
  std::uniform_int_distribution<> distrib(0, npart_cell_max);

  for (auto &ix : h_npart_cell) {
    ix = distrib(rng);
    ix += floor_occupancy;
  }
  const int min_occupancy =
      *std::min_element(h_npart_cell.begin(), h_npart_cell.end());
  const int max_occupancy =
      *std::max_element(h_npart_cell.begin(), h_npart_cell.end());
  ASSERT_TRUE(min_occupancy >= floor_occupancy);
  ASSERT_TRUE(max_occupancy <= npart_cell_max + floor_occupancy);

  ParticleLoopIterationSet iteration_set(nbin, ncell, h_npart_cell.data());

  auto is = iteration_set.get(std::nullopt, local_size);

  const int layer_offset = is.layer_offset;
  ASSERT_TRUE(layer_offset % local_size == 0);
  const int nbin_is = is.bin_end - is.bin_start;
  ASSERT_EQ(nbin_is, nbin);
  auto nd_ranges = is.nd_ranges;
  auto cell_offsets = is.cell_offsets;

  std::set<std::tuple<int, int>> correct;
  std::set<std::tuple<int, int>> to_test;

  for (int cellx = 0; cellx < ncell; cellx++) {
    for (int layerx = 0; layerx < h_npart_cell.at(cellx); layerx++) {
      correct.insert({cellx, layerx});
    }
  }

  // check the first bin
  auto nd_range0 = nd_ranges.at(0);
  auto local_range0 = nd_range0.get_local_range();
  auto local_range_size = local_range0.get(1);
  ASSERT_TRUE(local_range_size >= 1);
  ASSERT_TRUE(local_range_size <= local_size);

  auto global_range0 = nd_range0.get_global_range();
  auto cell_width = global_range0.get(0);
  auto layer_width = global_range0.get(1);

  for (int cx = 0; cx < cell_width; cx++) {
    for (int layerx = 0; layerx < layer_width; layerx++) {
      to_test.insert({cx, layerx});
    }
  }

  for (int binx = is.bin_start; binx < is.bin_end; binx++) {
    auto nd_range = nd_ranges.at(binx);
    auto local_range = nd_range.get_local_range();
    auto local_range_size = local_range.get(1);
    auto cell_offset = cell_offsets.at(binx);

    auto global_range = nd_range.get_global_range();
    auto cell_width = global_range.get(0);
    auto layer_width = global_range.get(1);

    for (int cx = 0; cx < cell_width; cx++) {
      for (int layerx = 0; layerx < layer_width; layerx++) {
        const int layer = layerx + layer_offset;
        const int cell = cx + cell_offset;
        if (layer < h_npart_cell.at(cell)) {
          to_test.insert({cell, layer});
        }
      }
    }
  }

  ASSERT_EQ(to_test, correct);
}

TEST(ParticleLoopIterationSet, zero) {

  const int nbin = 16;
  const int ncell = 1391;
  const int npart_cell_max = 89;
  const int local_size = 32;

  std::vector<int> h_npart_cell(ncell);

  std::mt19937 rng(522342);
  std::uniform_int_distribution<> distrib(0, npart_cell_max);

  for (auto &ix : h_npart_cell) {
    ix = distrib(rng);
  }

  for (int ix = ncell - 500; ix < ncell; ix++) {
    h_npart_cell.at(ix) = 0;
  }

  const int min_occupancy =
      *std::min_element(h_npart_cell.begin(), h_npart_cell.end());
  ASSERT_EQ(min_occupancy, 0);

  ParticleLoopIterationSet iteration_set(nbin, ncell, h_npart_cell.data());

  auto is = iteration_set.get(std::nullopt, local_size);

  const int layer_offset = is.layer_offset;
  ASSERT_TRUE(layer_offset % local_size == 0);
  const int nbin_is = is.bin_end - is.bin_start;
  auto nd_ranges = is.nd_ranges;
  auto cell_offsets = is.cell_offsets;

  std::set<std::tuple<int, int>> correct;
  std::set<std::tuple<int, int>> to_test;

  for (int cellx = 0; cellx < ncell; cellx++) {
    for (int layerx = 0; layerx < h_npart_cell.at(cellx); layerx++) {
      correct.insert({cellx, layerx});
    }
  }

  // check the first bin
  auto nd_range0 = nd_ranges.at(0);
  auto local_range0 = nd_range0.get_local_range();
  auto local_range_size = local_range0.get(1);
  ASSERT_TRUE(local_range_size >= 1);
  ASSERT_TRUE(local_range_size <= local_size);

  auto global_range0 = nd_range0.get_global_range();
  auto cell_width = global_range0.get(0);
  auto layer_width = global_range0.get(1);

  if (layer_offset) {
    for (int cx = 0; cx < cell_width; cx++) {
      for (int lx = 0; lx < layer_width; lx++) {
        to_test.insert({cx, lx});
      }
    }
  }

  for (int binx = is.bin_start; binx < is.bin_end; binx++) {
    auto nd_range = nd_ranges.at(binx);
    auto local_range = nd_range.get_local_range();
    auto local_range_size = local_range.get(1);
    auto cell_offset = cell_offsets.at(binx);

    auto global_range = nd_range.get_global_range();
    auto cell_width = global_range.get(0);
    auto layer_width = global_range.get(1);

    for (int cx = 0; cx < cell_width; cx++) {
      for (int lx = 0; lx < layer_width; lx++) {
        const int layer = lx + layer_offset;
        const int cell = cx + cell_offset;
        if (layer < h_npart_cell.at(cell)) {
          to_test.insert({cell, layer});
        }
      }
    }
  }

  ASSERT_EQ(to_test, correct);
}

TEST(ParticleLoopIterationSet, single_cell) {

  const int nbin = 16;
  const int ncell = 1391;
  const int npart_cell_max = 89;
  const int local_size = 32;

  std::vector<int> h_npart_cell(ncell);

  std::mt19937 rng(522342);
  std::uniform_int_distribution<> distrib(0, npart_cell_max);

  for (auto &ix : h_npart_cell) {
    ix = distrib(rng);
  }

  for (int ix = ncell - 500; ix < ncell; ix++) {
    h_npart_cell.at(ix) = 0;
  }

  const int min_occupancy =
      *std::min_element(h_npart_cell.begin(), h_npart_cell.end());
  ASSERT_EQ(min_occupancy, 0);

  ParticleLoopIterationSet iteration_set(nbin, ncell, h_npart_cell.data());

  for (int test_cell = 0; test_cell < ncell; test_cell++) {

    auto is = iteration_set.get(test_cell, local_size);

    const int layer_offset = is.layer_offset;
    auto nd_ranges = is.nd_ranges;
    auto cell_offsets = is.cell_offsets;

    std::set<std::tuple<int, int>> correct;
    std::set<std::tuple<int, int>> to_test;

    for (int layerx = 0; layerx < h_npart_cell.at(test_cell); layerx++) {
      correct.insert({test_cell, layerx});
    }

    // check the first bin

    if (layer_offset) {
      auto nd_range0 = nd_ranges.at(0);
      auto local_range0 = nd_range0.get_local_range();
      auto local_range_size = local_range0.get(1);
      ASSERT_TRUE(local_range_size >= 1);
      ASSERT_TRUE(local_range_size <= local_size);

      auto global_range0 = nd_range0.get_global_range();
      auto cell_width = global_range0.get(0);
      auto layer_width = global_range0.get(1);
      auto cell_offset = is.cell_offsets.at(0);

      for (int cx = 0; cx < cell_width; cx++) {
        for (int lx = 0; lx < layer_width; lx++) {
          to_test.insert({cx + cell_offset, lx});
        }
      }
    }

    for (int binx = is.bin_start; binx < is.bin_end; binx++) {
      auto nd_range = nd_ranges.at(binx);
      auto local_range = nd_range.get_local_range();
      auto local_range_size = local_range.get(1);
      auto cell_offset = cell_offsets.at(binx);

      auto global_range = nd_range.get_global_range();
      auto cell_width = global_range.get(0);
      auto layer_width = global_range.get(1);

      for (int cx = 0; cx < cell_width; cx++) {
        for (int lx = 0; lx < layer_width; lx++) {
          const int layer = lx + layer_offset;
          const int cell = cx + cell_offset;
          if (layer < h_npart_cell.at(cell)) {
            to_test.insert({cell, layer});
          }
        }
      }
    }

    ASSERT_EQ(to_test, correct);
  }
}
