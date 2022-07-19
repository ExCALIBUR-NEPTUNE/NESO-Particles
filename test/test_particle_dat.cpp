#include <CL/sycl.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <list>
#include <map>
#include <neso_particles.hpp>
#include <random>
#include <set>

using namespace NESO::Particles;

TEST(ParticleDat, test_particle_dat_append_1) {

  SYCLTarget sycl_target{GPU_SELECTOR, MPI_COMM_WORLD};

  const int cell_count = 4;
  const int ncomp = 3;

  auto A = ParticleDat(sycl_target, ParticleProp(Sym<INT>("BAR"), ncomp),
                       cell_count);

  std::vector<INT> counts(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_TRUE(A->h_npart_cell[cellx] == 0);
    counts[cellx] = 0;
  }

  const int N = 42;

  std::vector<INT> cells0(N);
  std::vector<INT> data0(N * ncomp);

  std::mt19937 rng(523905);
  std::uniform_int_distribution<int> cell_rng(0, cell_count - 1);

  std::vector<INT> layers0(N);

  INT index = 0;
  for (int px = 0; px < N; px++) {
    auto cell = cell_rng(rng);
    cells0[px] = cell;
    layers0[px] = counts[cell];
    counts[cell]++;
    for (int cx = 0; cx < ncomp; cx++) {
      data0[cx * N + px] = ++index;
    }
  }

  A->realloc(counts);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_TRUE(A->cell_dat.nrow[cellx] >= counts[cellx]);
  }

  A->append_particle_data(N, false, cells0, layers0, data0);
  // the append is async
  sycl_target.queue.wait();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_TRUE(A->h_npart_cell[cellx] == counts[cellx]);
    // the "data exists" flag is false so these new values should all be
    // zero
    auto cell_data = A->cell_dat.get_cell(cellx);
    for (int cx = 0; cx < ncomp; cx++) {
      for (int px = 0; px < (A->h_npart_cell[cellx]); px++) {
        ASSERT_TRUE((*cell_data)[cx][px] == 0);
      }
    }
  }

  // append the same counts with actual data
  for (int cellx = 0; cellx < cell_count; cellx++) {
    A->set_npart_cell(cellx, 0);
  }
  A->realloc(counts);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    ASSERT_TRUE(A->cell_dat.nrow[cellx] >= counts[cellx]);
  }

  A->append_particle_data(N, true, cells0, layers0, data0);
  // the append is async
  sycl_target.queue.wait();

  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto cell_data = A->cell_dat.get_cell(cellx);
    ASSERT_EQ(cell_data->nrow, counts[cellx]);
    ASSERT_EQ(cell_data->ncol, ncomp);

    for (int rowx = 0; rowx < cell_data->nrow; rowx++) {
      int row = -1;
      for (int rx = 0; rx < N; rx++) {
        if (data0[rx] == cell_data->data[0][rowx]) {
          row = rx;
          break;
        }
      }
      ASSERT_TRUE(row >= 0);

      for (int cx = 0; cx < ncomp; cx++) {
        ASSERT_EQ(cell_data->data[cx][rowx], data0[N * cx + row]);
      }

      ASSERT_EQ(cellx, cells0[row]);
      ASSERT_EQ(rowx, layers0[row]);
    }
  }
}
