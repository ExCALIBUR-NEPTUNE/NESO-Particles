#include "include/test_neso_particles.hpp"

namespace {

template <typename T> class TestGlobalArray : public GlobalArray<T> {
public:
  TestGlobalArray(SYCLTargetSharedPtr sycl_target, const std::size_t size,
                  const std::optional<T> init_value = std::nullopt)
      : GlobalArray<T>(sycl_target, size, init_value) {}

  inline void test_reduction_set(std::vector<T> &data) {
    this->impl_pre_loop_add();
    T *d_ptr = this->d_stage_buffer->ptr;
    const std::size_t size_bytes = sizeof(T) * this->size;
    this->sycl_target->queue.memcpy(d_ptr, data.data(), size_bytes)
        .wait_and_throw();
  }

  inline void test_impl_post_loop_add() { this->impl_post_loop_add(); }
};

} // namespace

TEST(GlobalArray, init) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 6151;
  GlobalArray<int> g0(sycl_target, N, 42);

  auto d0 = g0.get();
  for (auto &dx : d0) {
    EXPECT_EQ(dx, 42);
  }
  sycl_target->free();
}

TEST(GlobalArray, kernel_add) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  const int N = 13;
  TestGlobalArray<int> g0(sycl_target, N, 0);
  auto d0 = g0.get();
  for (auto &dx : d0) {
    EXPECT_EQ(dx, 0);
  }
  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  for (int ix = 0; ix < N; ix++) {
    d0[ix] = rank * ix;
  }

  g0.test_reduction_set(d0);
  g0.test_impl_post_loop_add();

  auto d1 = g0.get();
  std::vector<int> correct(N);
  for (int ix = 0; ix < N; ix++) {
    correct[ix] = 0;
    for (int rx = 0; rx < size; rx++) {
      correct[ix] += rx * ix;
    }
  }

  for (int ix = 0; ix < N; ix++) {
    EXPECT_EQ(d1[ix], correct[ix]);
  }

  sycl_target->free();
}

TEST(GlobalArray, repeat_kernel_add) {
  auto [A, sycl_target, cell_count_t] = particle_loop_common_2d(27, 16, 32);

  auto g0 = std::make_shared<GlobalArray<int>>(sycl_target, 1, 0);
  g0->fill(0);

  int npart_local = A->get_npart_local();
  int npart_total = -1;
  MPICHK(MPI_Allreduce(&npart_local, &npart_total, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD));

  auto loop0 =
      particle_loop(A, [=](auto GA) { GA.add(0, 1); }, Access::add(g0));

  loop0->execute();
  ASSERT_EQ(g0->get().at(0), npart_total);
  loop0->execute();
  ASSERT_EQ(g0->get().at(0), 2 * npart_total);

  sycl_target->free();
  A->domain->mesh->free();
}
