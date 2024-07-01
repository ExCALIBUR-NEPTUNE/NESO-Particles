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
}
