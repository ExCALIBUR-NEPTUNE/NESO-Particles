#include <gtest/gtest.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

TEST(CellDat, test_cell_dat_const_1) {

  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);

  const int cell_count = 4;
  const int nrow = 3;
  const int ncol = 5;

  CellDatConst<INT> cdc(sycl_target, cell_count, nrow, ncol);

  ASSERT_TRUE(cdc.nrow == nrow);
  ASSERT_TRUE(cdc.ncol == ncol);

  // should have malloced a cell_count * nrow * ncol sized block
  std::vector<INT> test_data(cell_count * nrow * ncol);

  // create some data and manually place in dat
  const int stride = nrow * ncol;
  int index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        test_data[cellx * stride + colx * nrow + rowx] = ++index;
      }
    }
  }
  sycl_target->queue
      .memcpy(cdc.device_ptr(), test_data.data(),
              test_data.size() * sizeof(INT))
      .wait();

  index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    CellData<INT> cell_data = cdc.get_cell(cellx);
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        // check data returned from dat using user interface
        ASSERT_TRUE(cell_data->data[colx][rowx] == ++index);
        // change the data
        cell_data->data[colx][rowx] = 2 * cell_data->data[colx][rowx];
      }
    }
    // write the changed data back to the celldat through the user interface
    cdc.set_cell(cellx, cell_data);
  }

  // check celldat data is correct
  sycl_target->queue
      .memcpy(test_data.data(), cdc.device_ptr(),
              test_data.size() * sizeof(INT))
      .wait();
  index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        ASSERT_TRUE(test_data[cellx * stride + colx * nrow + rowx] ==
                    2 * (++index));
      }
    }
  }
}

TEST(CellDat, test_cell_dat_REAL_1) {

  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);

  const int cell_count = 4;
  const int ncol = 2;

  CellDat<REAL> ddc(sycl_target, cell_count, ncol);

  // set some row counts
  std::vector<INT> nrows_0(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int cellx_nrow = cellx + 1;
    nrows_0[cellx] = cellx_nrow;
    ddc.set_nrow(cellx, cellx_nrow);
    ddc.wait_set_nrow();

    // check enough space was allocated
    ASSERT_TRUE(ddc.nrow_alloc[cellx] >= nrows_0[cellx]);
    ASSERT_TRUE(ddc.nrow[cellx] == nrows_0[cellx]);
  }

  REAL index = 0.0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    CellData<REAL> cd = ddc.get_cell(cellx);

    const int nrow = ddc.nrow[cellx];
    ASSERT_TRUE(cd->nrow == nrow);
    ASSERT_TRUE(cd->ncol == ddc.ncol);

    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        cd->data[colx][rowx] = ++index;
      }
    }

    ddc.set_cell(cellx, cd);
  }

  // set some new row counts
  std::vector<INT> nrows_1(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int cellx_nrow = (cellx + 1) * 2;
    nrows_1[cellx] = cellx_nrow;
    ddc.set_nrow(cellx, cellx_nrow);
    ddc.wait_set_nrow();

    // check enough space was allocated
    ASSERT_TRUE(ddc.nrow_alloc[cellx] >= nrows_1[cellx]);
    ASSERT_TRUE(ddc.nrow[cellx] == nrows_1[cellx]);
  }

  index = 0.0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    CellData<REAL> cd = ddc.get_cell(cellx);

    const int nrow = ddc.nrow[cellx];
    ASSERT_TRUE(cd->nrow == nrow);
    ASSERT_TRUE(cd->ncol == ddc.ncol);

    const int nrow0 = nrows_0[cellx];

    for (int rowx = 0; rowx < nrow0; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        ASSERT_TRUE(cd->data[colx][rowx] == ++index);
      }
    }
  }
}

TEST(CellDat, test_cell_dat_INT_1) {

  auto sycl_target = std::make_shared<SYCLTarget>(GPU_SELECTOR, MPI_COMM_WORLD);

  const int cell_count = 4;
  const int ncol = 2;

  CellDat<INT> ddc(sycl_target, cell_count, ncol);

  // set some row counts
  std::vector<INT> nrows_0(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int cellx_nrow = cellx + 1;
    nrows_0[cellx] = cellx_nrow;
    ddc.set_nrow(cellx, cellx_nrow);
    ddc.wait_set_nrow();

    // check enough space was allocated
    ASSERT_TRUE(ddc.nrow_alloc[cellx] >= nrows_0[cellx]);
    ASSERT_TRUE(ddc.nrow[cellx] == nrows_0[cellx]);
  }

  INT index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    CellData<INT> cd = ddc.get_cell(cellx);

    const int nrow = ddc.nrow[cellx];
    ASSERT_TRUE(cd->nrow == nrow);
    ASSERT_TRUE(cd->ncol == ddc.ncol);

    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        cd->data[colx][rowx] = ++index;
      }
    }

    ddc.set_cell(cellx, cd);
  }

  // set some new row counts
  std::vector<INT> nrows_1(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    const int cellx_nrow = (cellx + 1) * 2;
    nrows_1[cellx] = cellx_nrow;
    ddc.set_nrow(cellx, cellx_nrow);
    ddc.wait_set_nrow();

    // check enough space was allocated
    ASSERT_TRUE(ddc.nrow_alloc[cellx] >= nrows_1[cellx]);
    ASSERT_TRUE(ddc.nrow[cellx] == nrows_1[cellx]);
  }

  index = 0;
  for (int cellx = 0; cellx < cell_count; cellx++) {
    CellData<INT> cd = ddc.get_cell(cellx);

    const int nrow = ddc.nrow[cellx];
    ASSERT_TRUE(cd->nrow == nrow);
    ASSERT_TRUE(cd->ncol == ddc.ncol);

    const int nrow0 = nrows_0[cellx];

    for (int rowx = 0; rowx < nrow0; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        ASSERT_TRUE(cd->data[colx][rowx] == ++index);
      }
    }
  }
}

TEST(CellDatConst, fill) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int cell_count = 4;
  const int nrow = 2;
  const int ncol = 2;

  auto cdc =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, nrow, ncol);
  const INT value = 7;

  cdc->fill(value);

  for (int cx = 0; cx < cell_count; cx++) {
    auto cell_data = cdc->get_cell(cx);
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        EXPECT_EQ(cell_data->at(rowx, colx), value);
      }
    }
  }
}

TEST(CellDat, get_set_value) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int cell_count = 4;
  const int nrow = 3;
  const int ncol = 2;

  auto cd = std::make_shared<CellDat<INT>>(sycl_target, cell_count, ncol);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    cd->set_nrow(cellx, nrow);
    cd->wait_set_nrow();
  }

  INT index = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        index++;
        cd->set_value(cx, rowx, colx, index);
      }
    }
  }
  index = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        index++;
        const INT to_test = cd->get_value(cx, rowx, colx);
        ASSERT_EQ(to_test, index);
      }
    }
  }
}

TEST(CellDatConst, get_set_value) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int cell_count = 4;
  const int nrow = 3;
  const int ncol = 2;

  auto cdc =
      std::make_shared<CellDatConst<INT>>(sycl_target, cell_count, nrow, ncol);

  INT index = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        index++;
        cdc->set_value(cx, rowx, colx, index);
      }
    }
  }
  index = 0;
  for (int cx = 0; cx < cell_count; cx++) {
    for (int rowx = 0; rowx < nrow; rowx++) {
      for (int colx = 0; colx < ncol; colx++) {
        index++;
        const INT to_test = cdc->get_value(cx, rowx, colx);
        ASSERT_EQ(to_test, index);
      }
    }
  }
}
