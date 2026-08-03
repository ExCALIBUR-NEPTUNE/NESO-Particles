#include "include/test_neso_particles.hpp"

TEST(Algorithms, nd_local_array_loop_element_wise) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const int s0 = 511;
  const int s1 = 11;
  const int s2 = 7;

  auto a0 = std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, s0, s1, s2);
  auto a1 = std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, s0, s1, s2);
  auto a2 = std::make_shared<NDLocalArray<REAL, 3>>(sycl_target, s0, s1, s2);

  auto h0 = a0->get();
  auto h1 = a1->get();
  auto h2 = a2->get();

  std::size_t s = a0->index.size();

  REAL ii = 1.1;
  for (std::size_t ix = 0; ix < s; ix++) {
    h1.at(ix) = ii++;
    h2.at(ix) = ii++;
  }

  a0->fill(-1);
  a1->set(h1);
  a2->set(h2);

  nd_local_array_loop_element_wise(a0, Kernel::plus<REAL>{}, a1, a2);
  h0 = a0->get();

  for (std::size_t ix = 0; ix < s; ix++) {

    const REAL err = relative_error(h1.at(ix) + h2.at(ix), h0.at(ix));

    ASSERT_TRUE(err < 1.0e-14);
  }

  a0->fill(-1.0);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  nd_local_array_loop_element_wise(
      a0,
      [=](REAL a, REAL b, REAL c) -> REAL {
        NESO_KERNEL_ASSERT(a == -1.0, k_ep);
        return a + b / c;
      },
      a0, a1, a2);

  ASSERT_FALSE(ep.get_flag());

  h0 = a0->get();
  for (std::size_t ix = 0; ix < s; ix++) {

    const REAL err = relative_error(-1.0 + h1.at(ix) / h2.at(ix), h0.at(ix));

    ASSERT_TRUE(err < 1.0e-14);
  }

  sycl_target->free();
}
