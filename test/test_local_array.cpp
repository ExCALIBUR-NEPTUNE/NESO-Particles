#include "include/test_neso_particles.hpp"

TEST(LocalArray, init) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 6151;
  std::vector<int> d0(N);
  LocalArray<int> l0(sycl_target, N, 42);

  l0.get(d0);
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(42, d0[ix]);
  }

  sycl_target->free();
}

TEST(LocalArray, get_set) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 6151;
  std::vector<REAL> d0(N);
  std::iota(d0.begin(), d0.end(), 0);
  LocalArray<REAL> l0(sycl_target, d0);
  std::fill(d0.begin(), d0.end(), 0);
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(0, d0[ix]);
  }
  l0.get(d0);
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(ix, d0[ix]);
    d0[ix] *= 2;
  }
  l0.set(d0);
  std::fill(d0.begin(), d0.end(), 0);
  l0.get(d0);
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(ix * 2, d0[ix]);
  }

  LocalArray<int> l2(sycl_target, N, 43);
  auto d3 = l2.get();
  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(43, d3[ix]);
  }

  sycl_target->free();
}

TEST(NDLocalArray, get_set_nd_host_array) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto ndla_real =
      std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, 5, 3, 2);

  auto h_real = ndla_real->get();

  REAL ii = 1.0;
  for (auto &ix : h_real) {
    ix = ii++;
  }

  ndla_real->set(h_real);

  NDHostArraySharedPtr<REAL, 3> ndha_real = nullptr;
  ndla_real->get(ndha_real);

  ASSERT_NE(ndha_real, nullptr);

  ii = 1.0;
  for (int ix = 0; ix < (5 * 3 * 2); ix++) {
    ASSERT_EQ(ndha_real->ptr()[ix], ii++);
    ndha_real->ptr()[ix] = static_cast<REAL>(ix * 2);
  }

  ndla_real->set(ndha_real);

  h_real = ndla_real->get();
  for (int ix = 0; ix < (5 * 3 * 2); ix++) {
    ASSERT_EQ(h_real.at(ix), static_cast<REAL>(ix * 2));
  }

  sycl_target->free();
}

TEST(NDLocalArray, combine) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto r0 = std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, 5, 3, 2);
  auto r1 = std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, 5, 3, 2);

  auto hc = r0->get();
  auto h0 = r0->get();
  auto h1 = r1->get();

  REAL ii = 0.1;
  for (int ix = 0; ix < (5 * 3 * 2); ix++) {
    h0.at(ix) = ii++;
    h1.at(ix) = ii++;
    hc.at(ix) = Kernel::plus<REAL>{}(h0.at(ix), h1.at(ix));
  }

  r0->set(h0);
  r1->set(h1);

  r0->combine(r1, Kernel::plus<REAL>{});

  auto ht = r0->get();
  for (int ix = 0; ix < (5 * 3 * 2); ix++) {
    ASSERT_TRUE(relative_error(hc.at(ix), ht.at(ix)) < 1.0e-14);
  }

  sycl_target->free();
}

TEST(NDLocalArray, nd_host_array) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  auto a0 = std::make_shared<NDHostArray<int, 2>>(sycl_target, 10, 2);

  std::vector<int> h0;
  a0->get(h0);
  std::iota(h0.begin(), h0.end(), 1);
  a0->set(h0);

  auto d0 = std::make_shared<NDLocalArray<int, 2>>(sycl_target, a0);

  std::fill(h0.begin(), h0.end(), 0);

  a0->set(h0);
  d0->get(a0);
  a0->get(h0);

  for (int ix = 0; ix < 20; ix++) {
    ASSERT_EQ(h0.at(ix), ix + 1);
  }

  auto a1 = std::make_shared<NDHostArray<int, 2>>(sycl_target, d0);
  std::fill(h0.begin(), h0.end(), 0);
  a1->get(h0);

  for (int ix = 0; ix < 20; ix++) {
    ASSERT_EQ(h0.at(ix), ix + 1);
  }

  sycl_target->free();
}
