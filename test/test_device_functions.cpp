#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

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
