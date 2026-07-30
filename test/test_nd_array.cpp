#include "include/test_neso_particles.hpp"

TEST(NDHostArray, base) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto i0 = nd_index<1>(1);
  auto i1 = nd_index<2>(1, 2);
  auto i2 = nd_index<2>(1, 2);

  ASSERT_FALSE(i0 == i1);
  ASSERT_TRUE(i1 == i2);

  [[maybe_unused]] auto a1 = nd_host_array<int, 1>(sycl_target, 0);

  auto a2 = nd_host_array<int, 1>(sycl_target, 8);
  auto a3 = nd_host_array<int, 1>(sycl_target, 8);

  ASSERT_EQ(a2->index.size(), 8);

  for (int ix = 0; ix < 8; ix++) {
    a2->at(ix) = ix;
  }

  for (int ix = 0; ix < 8; ix++) {
    ASSERT_EQ(a2->at(ix), ix);
  }

  std::vector<int> h_a2;
  a2->get(h_a2);
  ASSERT_EQ(h_a2.size(), 8);

  for (int ix = 0; ix < 8; ix++) {
    ASSERT_EQ(h_a2.at(ix), ix);
    h_a2.at(ix) = 2 * ix;
  }

  a2->set(h_a2);

  for (int ix = 0; ix < 8; ix++) {
    ASSERT_EQ(a2->at(ix), 2 * ix);
  }

  a2->fill(-1);
  for (int ix = 0; ix < 8; ix++) {
    ASSERT_EQ(a2->at(ix), -1);
  }

  int *ptr = a2->ptr();
  for (int ix = 0; ix < 8; ix++) {
    ASSERT_EQ(ptr[ix], -1);
  }

  for (int ix = 0; ix < 8; ix++) {
    h_a2[ix] = -3 * ix;
  }

  a3->set(h_a2);
  a2->set(a3);
  for (int ix = 0; ix < 8; ix++) {
    ASSERT_EQ(ptr[ix], -3 * ix);
    ptr[ix] = 4 * ix;
  }

  a2->get(a3);
  for (int ix = 0; ix < 8; ix++) {
    ASSERT_EQ(a3->at(ix), 4 * ix);
  }

  sycl_target->free();
}
