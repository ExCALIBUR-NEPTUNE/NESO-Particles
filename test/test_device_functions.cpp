#include "include/test_neso_particles.hpp"

TEST(DeviceFunctions, consistent_line_orientation_2d) {

  REAL ax, ay, bx, by;
  REAL ix, iy, jx, jy;
  {
    ax = 1.0;
    ay = 2.0;
    bx = 3.0;
    by = 4.0;
    consistent_line_orientation_2d(ax, ay, bx, by, &ix, &iy, &jx, &jy);
    ASSERT_EQ(ax, ix);
    ASSERT_EQ(ay, iy);
    ASSERT_EQ(bx, jx);
    ASSERT_EQ(by, jy);
    consistent_line_orientation_2d(bx, by, ax, ay, &ix, &iy, &jx, &jy);
    ASSERT_EQ(ax, ix);
    ASSERT_EQ(ay, iy);
    ASSERT_EQ(bx, jx);
    ASSERT_EQ(by, jy);
  }

  {
    ax = 1.0;
    ay = 4.0;
    bx = 1.0;
    by = 2.0;
    consistent_line_orientation_2d(ax, ay, bx, by, &ix, &iy, &jx, &jy);
    ASSERT_EQ(bx, ix);
    ASSERT_EQ(by, iy);
    ASSERT_EQ(ax, jx);
    ASSERT_EQ(ay, jy);
    consistent_line_orientation_2d(bx, by, ax, ay, &ix, &iy, &jx, &jy);
    ASSERT_EQ(bx, ix);
    ASSERT_EQ(by, iy);
    ASSERT_EQ(ax, jx);
    ASSERT_EQ(ay, jy);
  }

  {
    ax = 1.0;
    ay = 2.0;
    bx = 3.0;
    by = 2.0;
    consistent_line_orientation_2d(ax, ay, bx, by, &ix, &iy, &jx, &jy);
    ASSERT_EQ(ax, ix);
    ASSERT_EQ(ay, iy);
    ASSERT_EQ(bx, jx);
    ASSERT_EQ(by, jy);
    consistent_line_orientation_2d(bx, by, ax, ay, &ix, &iy, &jx, &jy);
    ASSERT_EQ(ax, ix);
    ASSERT_EQ(ay, iy);
    ASSERT_EQ(bx, jx);
    ASSERT_EQ(by, jy);
  }
}

TEST(DeviceFunctions, line_segment_intersection_inner) {

  REAL l0, l1;
  line_segment_intersection_2d_lambda(1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0,
                                      l0, l1);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);

  line_segment_intersection_2d_lambda(1.0, 1.0, 1.0, 2.0, -1.0, 1.0, 1.0, 1.0,
                                      l0, l1);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);
  ASSERT_NEAR(l1, 1.0, 1.0e-15);

  line_segment_intersection_2d_lambda(1.0, 1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0,
                                      l0, l1);
  ASSERT_NEAR(l0, 1.0, 1.0e-15);
  ASSERT_NEAR(l1, 0.0, 1.0e-15);

  line_segment_intersection_2d_lambda(1.0, 1.0, 1.0, 2.0, -1.0, 2.0, 1.0, 2.0,
                                      l0, l1);
  ASSERT_NEAR(l0, 1.0, 1.0e-15);
  ASSERT_NEAR(l1, 1.0, 1.0e-15);

  line_segment_intersection_2d_lambda(1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0,
                                      l0, l1);
  ASSERT_NEAR(l0, 0.5, 1.0e-15);
  ASSERT_NEAR(l1, 0.5, 1.0e-15);

  line_segment_intersection_2d_lambda(2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0,
                                      l0, l1);
  ASSERT_NEAR(l0, 0.5, 1.0e-15);
  ASSERT_NEAR(l1, 0.5, 1.0e-15);
}

TEST(DeviceFunctions, line_segment_intersection) {
  REAL xi, yi;
  REAL l0 = -100.0;
  bool e;

  e = line_segment_intersection_2d(0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, xi,
                                   yi, l0);
  ASSERT_TRUE(!e);

  e = line_segment_intersection_2d(1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, xi,
                                   yi, l0);

  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, 1.0, 1.0e-15);
  ASSERT_NEAR(yi, 1.0, 1.0e-15);
  ASSERT_NEAR(l0, 0.0, 1.0e-15);

  e = line_segment_intersection_2d(1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, xi,
                                   yi, l0);

  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, 1.0, 1.0e-15);
  ASSERT_NEAR(yi, 1.0, 1.0e-15);

  e = line_segment_intersection_2d(1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, xi,
                                   yi, l0);

  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, 1.5, 1.0e-15);
  ASSERT_NEAR(yi, 1.5, 1.0e-15);

  e = line_segment_intersection_2d(0.0, 1.5, 2.0, 1.5, 1.0, 1.0, 1.0, 2.0, xi,
                                   yi, l0);

  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, 1.0, 1.0e-15);
  ASSERT_NEAR(yi, 1.5, 1.0e-15);
  ASSERT_NEAR(l0, 0.5, 1.0e-15);

  e = line_segment_intersection_2d(1.0, 1.0, 2.0, 2.0, 2.0, 1.5, 3.0, 1.5, xi,
                                   yi, l0);

  ASSERT_TRUE(!e);

  e = line_segment_intersection_2d(0.345521, 7.12994, 0.345521, 8.12994, 0.0,
                                   8.0, 1.0, 8.0, xi, yi, l0);
  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, 0.345521, 1.0e-15);
  ASSERT_NEAR(yi, 8.0, 1.0e-15);
  ASSERT_NEAR(l0, 8.0 - 7.12994, 1.0e-15);

  e = line_segment_intersection_2d(0.5, 7.5, 0.5, 8.5, 0.0, 8.0, 1.0, 8.0, xi,
                                   yi, l0);
  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, 0.5, 1.0e-15);
  ASSERT_NEAR(yi, 8.0, 1.0e-15);
  ASSERT_NEAR(l0, 0.5, 1.0e-15);

  e = line_segment_intersection_2d(0.5, 1.5, -0.5, 0.5, 0.0, 2.0, 0.0, -2.0, xi,
                                   yi, l0);
  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, 0.0, 1.0e-15);
  ASSERT_NEAR(yi, 1.0, 1.0e-15);
  ASSERT_NEAR(l0, 0.5, 1.0e-15);

  e = line_segment_intersection_2d(
      2.752242878045763e-12, 1, -0.09999999999750342, 1, -0.003183203807578749,
      -0.8488220999007776, -0.003183203807578749, 3.151177900099222, xi, yi,
      l0);
  ASSERT_TRUE(e);
  ASSERT_NEAR(xi, -0.003183203807578749, 1.0e-15);
  ASSERT_NEAR(yi, 1.0, 1.0e-15);
}

TEST(DeviceFunctions, matrix_invert_3) {

  const REAL M[9] = {
      0.6210163141855796, 0.3190160150713188, 0.2571805706556594,
      0.5936442375675933, 0.991744920278965,  0.6897101781042996,
      0.5889379079562244, 0.8198441731417937, 0.8429227586052362};

  const REAL Lcorrect[9] = {
      2.396111040073248,   -0.5142632694765471, -0.3102782441062093,
      -0.834399945091885,  3.2951384763344302,  -2.441622402938384,
      -0.8625733291639001, -2.8456117965337913, 3.777907865839368};

  REAL L[9];
  naive_matrix_inverse<3>(M, L);

  for (int ix = 0; ix < 9; ix++) {
    ASSERT_NEAR(L[ix], Lcorrect[ix], 1.0e-14);
  }
}

TEST(DeviceFunctions, matrix_invert_4) {

  const REAL M[16] = {
      0.0946422072387985, 0.8120429190962001, 0.4942494461121053,
      0.9178125430602081, 0.2850375256128153, 0.1463457132200754,
      0.4322885831714008, 0.4254072817319338, 0.7713729647584051,
      0.9431779836320303, 0.0572777823803825, 0.3918158930684735,
      0.1392738874243453, 0.1285321501053717, 0.8270943309693088,
      0.4699009249254635};

  const REAL Lcorrect[16] = {
      -1.0656931725086511, 2.527708148587856,  0.6244657934921137,
      -0.727545948845295,  0.4636364699948946, -3.6742381021991775,
      1.0168260293649087,  1.572901090240109,  -0.4912991088190402,
      -2.238254316420641,  0.4099585614121688, 2.64409330408957,
      1.0538003226407844,  4.1954832071005805, -1.1848051089178082,
      -2.7404814941081126};

  REAL L[16];
  naive_matrix_inverse<4>(M, L);

  for (int ix = 0; ix < 16; ix++) {
    ASSERT_NEAR(L[ix], Lcorrect[ix], 1.0e-14);
  }
}

TEST(Kernel, metadata) {
  {
    Kernel::Metadata metadata;

    EXPECT_EQ(metadata.num_bytes.value, 0);
    EXPECT_EQ(metadata.num_flops.value, 0);
  }

  {
    Kernel::Metadata metadata(Kernel::NumFLOP(256), Kernel::NumBytes(512));

    EXPECT_EQ(metadata.num_bytes.value, 512);
    EXPECT_EQ(metadata.num_flops.value, 256);
  }

  {
    Kernel::Metadata metadata(Kernel::NumBytes(123), Kernel::NumFLOP(5612));

    EXPECT_EQ(metadata.num_bytes.value, 123);
    EXPECT_EQ(metadata.num_flops.value, 5612);
  }
}

TEST(DeviceFunctions, line_segment_intersection_2d_x_axis_aligned) {

  {
    REAL xi, yi;
    ASSERT_TRUE(line_segment_intersection_2d_x_axis_aligned(
        5.0, 5.0, 5.0, 11.0, 0.0, 10.0, 10.0, xi, yi));
    ASSERT_NEAR(xi, 5.0, 1.0e-14);
    ASSERT_NEAR(yi, 10.0, 1.0e-14);
  }

  {
    REAL xi, yi;
    ASSERT_FALSE(line_segment_intersection_2d_x_axis_aligned(
        5.0, 5.0, 5.0, 9.0, 0.0, 10.0, 10.0, xi, yi));
  }

  {
    REAL xi, yi;
    ASSERT_FALSE(line_segment_intersection_2d_x_axis_aligned(
        5.0, 5.0, 4.0, 5.0, 0.0, 10.0, 10.0, xi, yi));
  }

  {
    REAL xi, yi;
    ASSERT_FALSE(line_segment_intersection_2d_x_axis_aligned(
        -2.0, 5.0, -2.0, -5.0, 0.0, -0.0, 10.0, xi, yi));
  }

  auto lambda_test = [&](REAL xa, REAL ya, REAL xb, REAL yb, REAL x0, REAL y0,
                         REAL x1) {
    const REAL y1 = y0;

    REAL xi_to_test, yi_to_test;
    ASSERT_TRUE(line_segment_intersection_2d_x_axis_aligned(
        xa, ya, xb, yb, x0, y0, x1, xi_to_test, yi_to_test));

    REAL xi_correct, yi_correct, l0;
    ASSERT_TRUE(line_segment_intersection_2d(
        xa, ya, xb, yb, x0, y0, x1, y1, xi_correct, yi_correct, l0, 1.0e-14));

    ASSERT_NEAR(xi_correct, xi_to_test, 1.0e-14);
    ASSERT_NEAR(yi_correct, yi_to_test, 1.0e-14);
  };

  lambda_test(5.0, 5.0, 5.0, 12.0, 0.0, 10.0, 10.0);

  lambda_test(5.0, 5.0, 5.0, -12.0, 0.0, 0.0, 10.0);

  lambda_test(5.0, 5.0, 2.0, -12.0, 0.0, 0.0, 10.0);
}

TEST(DeviceFunctions, line_segment_intersection_2d_y_axis_aligned) {

  auto lambda_test = [&](REAL xa, REAL ya, REAL xb, REAL yb, REAL x0, REAL y0,
                         REAL y1) {
    const REAL x1 = x0;

    REAL xi_to_test, yi_to_test;
    ASSERT_TRUE(line_segment_intersection_2d_y_axis_aligned(
        xa, ya, xb, yb, x0, y0, y1, xi_to_test, yi_to_test));

    REAL xi_correct, yi_correct, l0;
    ASSERT_TRUE(line_segment_intersection_2d(
        xa, ya, xb, yb, x0, y0, x1, y1, xi_correct, yi_correct, l0, 1.0e-14));

    ASSERT_NEAR(xi_correct, xi_to_test, 1.0e-14);
    ASSERT_NEAR(yi_correct, yi_to_test, 1.0e-14);
  };

  lambda_test(5.0, 5.0, -5.0, 5.0, 0.0, 0.0, 10.0);
  lambda_test(5.0, 5.0, 15.0, 5.0, 10.0, 0.0, 10.0);
}

TEST(DeviceFunctions, plane_intersection_3d_xy_plane_aligned) {

  auto lambda_test = [=](const REAL ax, const REAL ay, const REAL az,
                         const REAL bx, const REAL by, const REAL bz,
                         const REAL p0x, const REAL p0y, const REAL p0z,
                         const REAL p1x, const REAL p2y, const bool expected) {
    REAL xi_to_test;
    REAL yi_to_test;
    REAL zi_to_test;

    const bool contained_to_test = plane_intersection_3d_xy_plane_aligned(
        ax, ay, az, bx, by, bz, p0x, p0y, p0z, p1x, p2y, xi_to_test, yi_to_test,
        zi_to_test);

    REAL xi_correct;
    REAL yi_correct;
    REAL zi_correct;

    const bool contained_correct_x =
        line_segment_intersection_2d_x_axis_aligned(
            ax, az, bx, bz, p0x, p0z, p1x, xi_correct, zi_correct);
    const bool contained_correct_y =
        line_segment_intersection_2d_x_axis_aligned(
            ay, az, by, bz, p0y, p0z, p2y, yi_correct, zi_correct);

    ASSERT_EQ(contained_correct_x && contained_correct_y, contained_to_test);
    ASSERT_EQ(expected, contained_to_test);

    if (contained_to_test) {
      ASSERT_NEAR(xi_correct, xi_to_test, 1.0e-12);
      ASSERT_NEAR(yi_correct, yi_to_test, 1.0e-12);
      ASSERT_NEAR(zi_correct, zi_to_test, 1.0e-12);
    }
  };

  lambda_test(0.2, 0.3, 0.5, 0.2, 0.3, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0, true);

  lambda_test(0.2, 0.3, 0.5, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 1.0, 1.0, false);

  lambda_test(0.2, 0.3, 0.5, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, true);

  lambda_test(0.2, 0.3, 0.5, 0.2, 0.3, 0.0, 0.1, 0.2, 0.2, 1.1, 1.2, true);

  lambda_test(0.2, 0.3, -0.5, 0.7, 0.6, 0.4, 0.1, 0.2, 0.2, 1.1, 1.2, true);
}

TEST(DeviceFunctions, bitonic8) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int size = sycl_target->comm_pair.size_parent;
  const int rank = sycl_target->comm_pair.rank_parent;

  std::vector<int> s(8);
  std::iota(s.begin(), s.end(), 0);

  std::vector<int> t;
  const int num_permutations = 8 * 7 * 6 * 5 * 4 * 3 * 2;
  t.reserve(num_permutations / size);

  int rstart = 0;
  int rend = 0;
  get_decomp_1d(size, num_permutations, rank, &rstart, &rend);

  for (int px = 1; px < rstart; px++) {
    std::next_permutation(s.begin(), s.end());
  }
  for (int px = rstart; px < rend; px++) {
    t.insert(t.begin(), s.begin(), s.end());
    std::next_permutation(s.begin(), s.end());
  }

  ASSERT_EQ(t.size() % 8, 0);

  BufferDevice<int> d_t(sycl_target, t);

  std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  const std::size_t num_elements = t.size();
  local_size = std::max((std::size_t)8, (local_size / 8) * 8);
  std::size_t global_size = get_next_multiple(num_elements, local_size);
  int *k_t = d_t.ptr;

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        sycl::local_accessor<int> la(sycl::range<1>(local_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size),
                              sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> idx) {
              const std::size_t local_id = idx.get_local_linear_id();
              const std::size_t global_id = idx.get_global_linear_id();
              if (global_id < num_elements) {
                la[local_id] = k_t[global_id];
              }

              Kernel::bitonic8(idx.get_group(), &la[(local_id / 8) * 8]);

              if (global_id < num_elements) {
                k_t[global_id] = la[local_id];
              }
            });
      })
      .wait_and_throw();

  auto h_t = d_t.get();

  std::vector<int> u(8);
  std::iota(s.begin(), s.end(), 0);

  std::size_t index = 0;
  while (index < h_t.size()) {
    for (std::size_t ix = 0; ix < 8; ix++) {
      u.at(ix) = h_t.at(index + ix);
    }
    ASSERT_EQ(u, s);
    index += 8;
  }

  sycl_target->free();
}

TEST(DeviceFunctions, cross_product_3d) {
  const REAL a[3] = {1, 2, 3};
  const REAL b[3] = {3, 4, 5};
  REAL to_test[3] = {0, 0, 0};
  const REAL correct[3] = {-2, 4, -2};

  Kernel::cross_product(a[0], a[1], a[2], b[0], b[1], b[2], to_test,
                        to_test + 1, to_test + 2);

  ASSERT_EQ(correct[0], to_test[0]);
  ASSERT_EQ(correct[1], to_test[1]);
  ASSERT_EQ(correct[2], to_test[2]);
}

TEST(DeviceFunctions, joint_reduce) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  auto lambda_run_test = [&](auto to_test) {
    using value_type = typename decltype(to_test)::value_type;
    const auto correct = std::accumulate(to_test.begin(), to_test.end(),
                                         static_cast<value_type>(0));

    const std::size_t k_n = to_test.size();
    if (to_test.size() == 0) {
      to_test.resize(1);
    }

    std::vector<value_type> output = {0};
    BufferDevice d_output(sycl_target, output);
    BufferDevice d_to_test(sycl_target, to_test);

    auto *k_output = d_output.ptr;
    auto *k_to_test = d_to_test.ptr;

    for (std::size_t ls = 1; ls <= local_size; ls *= 2) {

      sycl_target->queue
          .parallel_for(
              sycl::nd_range<1>(sycl::range<1>(ls), sycl::range<1>(ls)),
              [=](auto idx) {
                auto v = Kernel::joint_reduce(idx.get_group(), k_to_test,
                                              k_to_test + k_n,
                                              Kernel::plus<value_type>{});

                if (idx.get_local_id(0) == 0) {
                  k_output[0] = v;
                }
              })
          .wait_and_throw();

      ASSERT_TRUE(relative_error(correct, d_output.get().at(0)) < 1.0e-8);
    }
  };

  std::mt19937 rng(5234234 + sycl_target->comm_pair.rank_parent);
  const int N = 18123;

  {
    std::vector<int> empty;
    lambda_run_test(empty);

    std::vector<int> small = {9, 2, 5};
    lambda_run_test(small);

    std::uniform_int_distribution<int> dist(-71, 72);
    std::vector<int> large(N);
    for (int ix = 0; ix < N; ix++) {
      large[ix] = dist(rng);
    }

    lambda_run_test(large);
  }
  {
    std::vector<INT> empty;
    lambda_run_test(empty);

    std::vector<INT> small = {9, 2, 5};
    lambda_run_test(small);

    std::uniform_int_distribution<INT> dist(-71, 72);
    std::vector<INT> large(N);
    for (int ix = 0; ix < N; ix++) {
      large[ix] = dist(rng);
    }

    lambda_run_test(large);
  }

  {
    std::vector<REAL> empty;
    lambda_run_test(empty);

    std::vector<REAL> small = {9, 2, 5};
    lambda_run_test(small);

    std::uniform_real_distribution<REAL> dist(-71, 72);
    std::vector<REAL> large(N);
    for (int ix = 0; ix < N; ix++) {
      large[ix] = dist(rng);
    }

    lambda_run_test(large);
  }

  sycl_target->free();
}

TEST(DeviceFunctions, div_round_up) {
  ASSERT_EQ(div_round_up(0, 1), 0);
  ASSERT_EQ(div_round_up(-1, 1), -1);
  ASSERT_EQ(div_round_up(-3, 1), -3);
  ASSERT_EQ(div_round_up(-3, 2), (-3) / 2);

  ASSERT_EQ(div_round_up(1, 1), 1);
  ASSERT_EQ(div_round_up(4, 2), 2);
  ASSERT_EQ(div_round_up(5, 2), 3);
}

TEST(DeviceFunctions, line_triangle_intersection_moller_trumbore) {

  sycl::marray<REAL, 3> line_origin{1.25, 1.25, -1.0};
  sycl::marray<REAL, 3> line_direction{0.0, 0.0, 1.0};
  sycl::marray<REAL, 3> v0{1.0, 1.0, 0.0};
  sycl::marray<REAL, 3> v1{2.0, 1.0, 0.0};
  sycl::marray<REAL, 3> v2{2.0, 2.0, 0.0};
  sycl::marray<REAL, 3> intersection_point{0.0, 0.0, 0.0};
  sycl::marray<REAL, 3> bary_coords{0.0, 0.0, 0.0};

  REAL t = 0.0;

  bool contained = line_triangle_intersection_moller_trumbore(
      line_origin, line_direction, v0, v1, v2, bary_coords, t);

  evaluate_barycentric_coordinates(bary_coords, v0, v1, v2, intersection_point);

  ASSERT_TRUE(contained);
  ASSERT_NEAR(intersection_point[0], 1.25, 1.0e-14);
  ASSERT_NEAR(intersection_point[1], 1.25, 1.0e-14);
  ASSERT_NEAR(intersection_point[2], 0.0, 1.0e-14);

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::vector<sycl::marray<REAL, 3>> h_test = {intersection_point};
  BufferDevice<sycl::marray<REAL, 3>> d_test(sycl_target, h_test);
  auto *k_test = d_test.ptr;

  BufferDevice<int> d_flag(sycl_target, std::vector<int>({0}));
  auto *k_flag = d_flag.ptr;

  sycl_target->queue
      .single_task([=]() {
        sycl::marray<REAL, 3> bary_coords{0.0, 0.0, 0.0};
        REAL t = 0.0;
        k_flag[0] = line_triangle_intersection_moller_trumbore(
            line_origin, line_direction, v0, v1, v2, bary_coords, t);
        evaluate_barycentric_coordinates(bary_coords, v0, v1, v2, *k_test);
      })
      .wait_and_throw();

  auto h_flag = d_flag.get();
  d_test.get(h_test);

  ASSERT_TRUE(h_flag[0]);
  ASSERT_NEAR(h_test[0][0], 1.25, 1.0e-14);
  ASSERT_NEAR(h_test[0][1], 1.25, 1.0e-14);
  ASSERT_NEAR(h_test[0][2], 0.0, 1.0e-14);

  sycl_target->free();

  std::mt19937 rng(5234234);
  const int num_samples = 100000;
  std::uniform_real_distribution<REAL> dist_bary(-0.2, 1.2);
  std::uniform_real_distribution<REAL> dist_direction(-1.0, 1.0);

  auto lambda_test_triangle = [&](const auto v0, const auto v1, const auto v2) {
    const sycl::marray<REAL, 3> E1 = v1 - v0;
    const sycl::marray<REAL, 3> E2 = v2 - v0;
    const sycl::marray<REAL, 3> normal = sycl::cross(E1, E2);
    sycl::marray<REAL, 3> to_test_intersection_point =
        sycl::marray<REAL, 3>(0.0, 0.0, 0.0);

    for (int testx = 0; testx < num_samples; testx++) {

      const REAL l1 = dist_bary(rng);
      const REAL l2 = dist_bary(rng);
      const REAL l0 = 1.0 - l1 - l2;

      const bool in_triangle = (l0 >= 0.0) && (l1 >= 0.0) && (l2 >= 0.0);

      sycl::marray<REAL, 3> correct_intersection_point =
          l0 * v0 + l1 * v1 + l2 * v2;

      sycl::marray<REAL, 3> direction_out{
          dist_direction(rng), dist_direction(rng), dist_direction(rng)};

      const bool in_plane = std::fabs(sycl::dot(normal, direction_out)) <= 0.0;

      const bool correct_contained = (!in_plane) && in_triangle;

      sycl::marray<REAL, 3> line_direction = -1 * direction_out;
      sycl::marray<REAL, 3> line_origin =
          correct_intersection_point + direction_out;

      const bool to_test_contained = line_triangle_intersection_moller_trumbore(
          line_origin, line_direction, v0, v1, v2, bary_coords, t, 1.0e-15,
          1.0e-15);

      evaluate_barycentric_coordinates(bary_coords, v0, v1, v2,
                                       to_test_intersection_point);

      const auto diff = correct_intersection_point - to_test_intersection_point;
      const REAL err = std::sqrt(sycl::dot(diff, diff));

      ASSERT_NEAR(err, 0.0, 1.0e-14);

      // If the test point is contained then the test with 1E-15 padding should
      // also be contained.
      if (correct_contained) {
        ASSERT_TRUE(to_test_contained);
      } else {

        const REAL ll0 = bary_coords[0];
        const REAL ll1 = bary_coords[1];
        const REAL ll2 = bary_coords[2];

        const REAL tol = 1.0e-14;

        const bool test_sum = ll0 + ll1 + ll2 <= 1.0 + tol;
        const bool test_ll0 = ll0 >= -tol;
        const bool test_ll1 = ll1 >= -tol;
        const bool test_ll2 = ll2 >= -tol;

        const bool near_enough = test_ll0 && test_ll1 && test_ll2 && test_sum;

        ASSERT_TRUE((!to_test_contained) || near_enough);
      }
    }
  };

  lambda_test_triangle(v0, v1, v2);
}

TEST(DeviceFunctions, reduce_over_group_block_wise) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  for (std::size_t block_size = 1; block_size <= local_size; block_size *= 2) {
    const std::size_t d1 = block_size;
    const std::size_t d0 = local_size / d1;
    ASSERT_EQ(d1 * d0, local_size);

    const std::size_t n0 = 23;
    const std::size_t n1 = 31;
    const std::size_t e0 = d0 * n0;
    const std::size_t e1 = d1 * n1;

    std::vector<int> h_source(e0 * e1);
    std::iota(h_source.begin(), h_source.end(), 0);

    BufferDevice<int> d_source(sycl_target, h_source);
    int *k_source = d_source.ptr;
    BufferDevice<int> d_dest(sycl_target, h_source);
    int *k_dest = d_dest.ptr;

    sycl_target->queue
        .submit([=](auto &cgh) {
          sycl::local_accessor<int, 1> local_memory(sycl::range<1>(local_size),
                                                    cgh);

          cgh.parallel_for(
              sycl::nd_range<2>(sycl::range<2>(e0, e1), sycl::range<2>(d0, d1)),
              [=](sycl::nd_item<2> idx) {
                const std::size_t gid = idx.get_global_linear_id();
                const std::size_t lid = idx.get_local_linear_id();
                local_memory[lid] = k_source[gid];

                const bool contributed = Kernel::reduce_over_group_block_wise(
                    &local_memory[0], idx, sycl::plus<int>{});

                k_dest[gid] = contributed ? local_memory[lid] : -1;
              });
        })
        .wait_and_throw();

    auto h_dest = d_dest.get();
    auto h_correct = d_dest.get();

    std::fill(h_correct.begin(), h_correct.end(), -2);

    for (std::size_t r0 = 0; r0 < e0; r0++) {
      for (std::size_t b1 = 0; b1 < n1; b1++) {

        const std::size_t start = r0 * e1 + b1 * d1;
        const std::size_t end = start + d1;

        int acc = 0;
        for (std::size_t ix = start; ix < end; ix++) {
          acc += h_source.at(ix);
          h_correct.at(ix) = -1;
        }
        h_correct.at(start) = acc;
      }
    }

    ASSERT_EQ(h_correct, h_dest);
  }

  sycl_target->free();
}
