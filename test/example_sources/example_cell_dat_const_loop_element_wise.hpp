// clang-format off
void cell_dat_const_loop_element_wise_example(SYCLTargetSharedPtr sycl_target){

  // Create CellDatConsts of the same size and shape.
  const int cell_count = 61;
  int nrow = 3;
  int ncol = 7;
  auto a =
    std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);
  auto b =
    std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);
  auto c =
    std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);
  auto d =
    std::make_shared<CellDatConst<REAL>>(sycl_target, cell_count, nrow, ncol);

  // Create some initial data for the arguments a,b and c.
  std::mt19937 rng(522342 + sycl_target->comm_pair.rank_parent);
  std::uniform_real_distribution<REAL> dist(1.0, 4.0);
  auto h_a = a->get_all_cells();
  auto h_b = b->get_all_cells();
  auto h_c = c->get_all_cells();

  for (int cellx = 0; cellx < cell_count; cellx++) {
    for (int colx = 0; colx < ncol; colx++) {
      for (int rowx = 0; rowx < nrow; rowx++) {
        const auto ta = dist(rng);
        const auto tb = dist(rng);
        const auto tc = dist(rng);
        h_a.at(cellx)->at(rowx, colx) = ta;
        h_b.at(cellx)->at(rowx, colx) = tb;
        h_c.at(cellx)->at(rowx, colx) = tc;
      }
    }
  }

  a->set_all_cells(h_a);
  b->set_all_cells(h_b);
  c->set_all_cells(h_c);
  d->fill(0);

  // d[cell, row, col] = 
  //    a[cell, row, col] * b[cell, row, col] + c[cell, row, col]
  //
  // Note that this is a scalar valued function of scalars that is applied 
  // element wise. This function should be device copyable and executable
  // on the compute device.
  cell_dat_const_loop_element_wise(
      d, [=](REAL a, REAL b, REAL c) -> REAL { return a * b + c; }, a, b, c);
}
// clang-format on
