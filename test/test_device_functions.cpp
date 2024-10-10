#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

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
    EXPECT_EQ(metadata.local_size.value, 0);
  }

  {
    Kernel::Metadata metadata(Kernel::LocalSize(256));

    EXPECT_EQ(metadata.num_bytes.value, 0);
    EXPECT_EQ(metadata.num_flops.value, 0);
    EXPECT_EQ(metadata.local_size.value, 256);
  }

  {
    Kernel::Metadata metadata(Kernel::NumFlops(256), Kernel::NumBytes(512));

    EXPECT_EQ(metadata.num_bytes.value, 512);
    EXPECT_EQ(metadata.num_flops.value, 256);
    EXPECT_EQ(metadata.local_size.value, 0);
  }

  {
    Kernel::Metadata metadata(Kernel::NumBytes(123), Kernel::NumFlops(5612),
                              Kernel::LocalSize(256));

    EXPECT_EQ(metadata.num_bytes.value, 123);
    EXPECT_EQ(metadata.num_flops.value, 5612);
    EXPECT_EQ(metadata.local_size.value, 256);
  }
}
