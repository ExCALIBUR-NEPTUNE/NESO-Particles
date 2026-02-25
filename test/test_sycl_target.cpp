#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace {

template <template <typename> typename SPEC, typename T>
inline bool atomic_binop_check_long(sycl::queue queue, const SPEC<T> spec) {
  constexpr std::size_t num_elements = 4096 * 4;
  constexpr std::size_t num_bytes = num_elements * sizeof(T);
  T *d_ptr = static_cast<T *>(sycl::malloc_device(num_bytes, queue));

  T h_correct = SPEC<T>::identity_element;
  T h_to_test = h_correct;

  T *d_result = static_cast<T *>(sycl::malloc_device(sizeof(T), queue));
  queue.memcpy(d_result, &h_correct, sizeof(T)).wait_and_throw();

  std::vector<T> elements(num_elements);
  std::random_device rng{};
  std::uniform_real_distribution<double> dist(static_cast<double>(-50),
                                              static_cast<double>(50));
  for (auto &ex : elements) {
    ex = static_cast<T>(dist(rng));
  }
  queue.memcpy(d_ptr, elements.data(), num_bytes).wait_and_throw();

  bool passed = true;
  for (std::size_t ix = 0; ix < num_elements; ix++) {
    queue.single_task<>([=]() { spec.binop_device(d_result, d_ptr[ix]); })
        .wait_and_throw();
    queue.memcpy(&h_to_test, d_result, sizeof(T)).wait_and_throw();
    spec.binop_host(&h_correct, elements.at(ix));
    passed = passed && spec.test(h_correct, h_to_test);
  }

  sycl::free(d_ptr, queue);
  sycl::free(d_result, queue);
  return passed;
}

} // namespace

TEST(SYCLTarget, print_device_info) {

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  bool to_test = false;
#ifdef NESO_PARTICLES
  to_test = true;
#endif
  ASSERT_TRUE(to_test);

  sycl_target->print_device_info();
  sycl_target->free();
}

TEST(SYCLTarget, joint_exclusive_scan_int) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  for (std::size_t N : {1, 2, 100, 21241}) {
    const std::size_t group_size =
        std::min(static_cast<std::size_t>(
                     sycl_target->device
                         .get_info<sycl::info::device::max_work_group_size>()),
                 static_cast<std::size_t>(N));
    ASSERT_TRUE(group_size >= 1);

    std::vector<int> h_src(N);
    std::vector<int> h_correct(N);
    std::iota(h_src.begin(), h_src.end(), 1);
    std::exclusive_scan(h_src.begin(), h_src.end(), h_correct.begin(), 0);

    BufferDevice d_src(sycl_target, h_src);
    BufferDevice d_dst(sycl_target, h_src);

    joint_exclusive_scan(sycl_target, N, d_src.ptr, d_dst.ptr).wait_and_throw();

    auto h_to_test = d_dst.get();

    EXPECT_EQ(h_to_test, h_correct);
  }

  sycl_target->free();
}

TEST(SYCLTarget, joint_exclusive_scan_INT) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t N = 21241;
  const std::size_t group_size =
      std::min(static_cast<std::size_t>(
                   sycl_target->device
                       .get_info<sycl::info::device::max_work_group_size>()),
               static_cast<std::size_t>(N));
  ASSERT_TRUE(group_size >= 1);

  std::vector<INT> h_src(N);
  std::vector<INT> h_correct(N);
  std::iota(h_src.begin(), h_src.end(), 1);
  std::exclusive_scan(h_src.begin(), h_src.end(), h_correct.begin(), 0);

  BufferDevice d_src(sycl_target, h_src);
  BufferDevice d_dst(sycl_target, h_src);

  joint_exclusive_scan(sycl_target, N, d_src.ptr, d_dst.ptr).wait_and_throw();

  auto h_to_test = d_dst.get();

  EXPECT_EQ(h_to_test, h_correct);

  sycl_target->free();
}

TEST(SYCLTarget, joint_exclusive_scan_n_int) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t num_arrays = 7;
  const std::size_t N = 2121;

  std::vector<int> h_src;
  std::vector<int> h_correct;
  std::vector<int> h_array_sizes(num_arrays);
  std::vector<int> h_array_offsets(num_arrays);

  std::mt19937 rng(52234234);
  std::uniform_int_distribution<int> dist{0, 512};

  h_src.reserve(num_arrays * N);
  h_correct.reserve(num_arrays * N);

  int offset = 0;
  for (std::size_t ax = 0; ax < num_arrays; ax++) {
    const std::size_t num_elements = (ax == 10) ? 0 : N - ax;
    std::vector<int> tmp_src(num_elements);
    std::vector<int> tmp_correct(num_elements);

    h_array_sizes[ax] = num_elements;
    h_array_offsets[ax] = offset;
    offset += num_elements;

    int current = 0;
    for (std::size_t ex = 0; ex < num_elements; ex++) {
      const int value = dist(rng);
      tmp_src[ex] = value;
      tmp_correct[ex] = current;
      current += value;
    }

    h_src.insert(h_src.end(), tmp_src.begin(), tmp_src.end());
    h_correct.insert(h_correct.end(), tmp_correct.begin(), tmp_correct.end());
  }

  BufferDevice d_src(sycl_target, h_src);
  BufferDevice d_dst(sycl_target, h_src);
  BufferDevice d_array_sizes(sycl_target, h_array_sizes);
  BufferDevice d_array_offsets(sycl_target, h_array_offsets);

  joint_exclusive_scan_n(sycl_target, num_arrays, d_array_sizes.ptr,
                         d_array_offsets.ptr, d_src.ptr, d_dst.ptr)
      .wait_and_throw();

  auto h_to_test = d_dst.get();
  EXPECT_EQ(h_to_test, h_correct);

  sycl_target->free();
}

TEST(SYCLTarget, joint_inclusive_scan_n_int) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t num_arrays = 7;
  const std::size_t N = 2121;

  std::vector<int> h_src;
  std::vector<int> h_correct;
  std::vector<int> h_array_sizes(num_arrays);
  std::vector<int> h_array_offsets(num_arrays);

  std::mt19937 rng(52234234);
  std::uniform_int_distribution<int> dist{0, 512};

  h_src.reserve(num_arrays * N);
  h_correct.reserve(num_arrays * N);

  int offset = 0;
  for (std::size_t ax = 0; ax < num_arrays; ax++) {
    const std::size_t num_elements = (ax == 10) ? 0 : N - ax;
    std::vector<int> tmp_src(num_elements);
    std::vector<int> tmp_correct(num_elements);

    h_array_sizes[ax] = num_elements;
    h_array_offsets[ax] = offset;
    offset += num_elements;

    int current = 0;
    for (std::size_t ex = 0; ex < num_elements; ex++) {
      const int value = dist(rng);
      current += value;
      tmp_src[ex] = value;
      tmp_correct[ex] = current;
    }

    h_src.insert(h_src.end(), tmp_src.begin(), tmp_src.end());
    h_correct.insert(h_correct.end(), tmp_correct.begin(), tmp_correct.end());
  }

  BufferDevice d_src(sycl_target, h_src);
  BufferDevice d_dst(sycl_target, h_src);
  BufferDevice d_array_sizes(sycl_target, h_array_sizes);
  BufferDevice d_array_offsets(sycl_target, h_array_offsets);

  joint_inclusive_scan_n(sycl_target, num_arrays, d_array_sizes.ptr,
                         d_array_offsets.ptr, d_src.ptr, d_dst.ptr)
      .wait_and_throw();

  auto h_to_test = d_dst.get();
  EXPECT_EQ(h_to_test, h_correct);

  sycl_target->free();
}

TEST(SYCLTarget, joint_exclusive_scan_n_sum_int) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t num_arrays = 355;
  const std::size_t N = 2120;

  std::vector<int> h_src;
  std::vector<int> h_correct;
  std::vector<int> h_totals;
  std::vector<int> h_array_sizes(num_arrays);
  std::vector<int> h_array_offsets(num_arrays);

  std::mt19937 rng(52234234);
  std::uniform_int_distribution<int> dist{0, 512};

  h_src.reserve(num_arrays * N);
  h_correct.reserve(num_arrays * N);
  h_totals.reserve(num_arrays);

  int offset = 0;
  for (std::size_t ax = 0; ax < num_arrays; ax++) {
    const std::size_t num_elements = (ax == 10) ? 0 : N - ax;
    std::vector<int> tmp_src(num_elements);
    std::vector<int> tmp_correct(num_elements);

    h_array_sizes[ax] = num_elements;
    h_array_offsets[ax] = offset;
    offset += num_elements;

    int current = 0;
    for (std::size_t ex = 0; ex < num_elements; ex++) {
      const int value = dist(rng);
      tmp_src[ex] = value;
      tmp_correct[ex] = current;
      current += value;
    }

    h_totals.push_back(current);
    h_src.insert(h_src.end(), tmp_src.begin(), tmp_src.end());
    h_correct.insert(h_correct.end(), tmp_correct.begin(), tmp_correct.end());
  }

  BufferDevice d_src(sycl_target, h_src);
  BufferDevice d_dst(sycl_target, h_src);
  BufferDevice d_array_sizes(sycl_target, h_array_sizes);
  BufferDevice d_array_offsets(sycl_target, h_array_offsets);
  BufferDevice<int> d_totals(sycl_target, static_cast<std::size_t>(num_arrays));

  joint_exclusive_scan_n_sum(sycl_target, num_arrays, d_array_sizes.ptr,
                             d_array_offsets.ptr, d_src.ptr, d_dst.ptr,
                             d_totals.ptr)
      .wait_and_throw();

  // check the exclusive scan
  auto h_to_test = d_dst.get();
  EXPECT_EQ(h_to_test, h_correct);
  // check the totals
  auto h_to_test_totals = d_totals.get();
  EXPECT_EQ(h_totals, h_to_test_totals);

  sycl_target->free();
}

TEST(SYCLTarget, parameters) {
  const std::size_t local_size =
      get_env_size_t("NESO_PARTICLES_LOOP_LOCAL_SIZE", 32);
  const std::size_t nbin = get_env_size_t("NESO_PARTICLES_LOOP_NBIN", 4);

  auto p_local_size = std::make_shared<SizeTParameter>(local_size);
  auto p_nbin = std::make_shared<SizeTParameter>(nbin);

  Parameters p;
  p.set("LOOP_LOCAL_SIZE", p_local_size);
  p.set("LOOP_NBIN", p_nbin);

  EXPECT_EQ(p.get<SizeTParameter>("LOOP_LOCAL_SIZE")->value, local_size);
  EXPECT_EQ(p.get<SizeTParameter>("LOOP_NBIN")->value, nbin);
}

TEST(SYCLTarget, matrix_transpose) {

  auto lambda_transpose = [](const std::size_t num_rows,
                             const std::size_t num_cols, auto &h_src,
                             auto &h_dst) {
    for (std::size_t rowx = 0; rowx < num_rows; rowx++) {
      for (std::size_t colx = 0; colx < num_cols; colx++) {
        h_dst.at(colx * num_rows + rowx) = h_src.at(rowx * num_cols + colx);
      }
    }
  };

  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  {
    std::vector<int> h_src_simple = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> h_dst_correct = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<int> h_dst_to_test(8);
    lambda_transpose(2, 4, h_src_simple, h_dst_to_test);
    ASSERT_EQ(h_dst_correct, h_dst_to_test);

    BufferDevice<int> d_src(sycl_target, h_src_simple);
    BufferDevice<int> d_dst(sycl_target, h_src_simple);

    matrix_transpose(sycl_target, 2, 4, d_src.ptr, d_dst.ptr).wait_and_throw();
    std::fill(h_dst_to_test.begin(), h_dst_to_test.end(), 0);
    d_dst.get(h_dst_to_test);
    ASSERT_EQ(h_dst_correct, h_dst_to_test);
  }

  {
    const std::size_t num_rows = 7919;
    const std::size_t num_cols = 1483;
    std::vector<REAL> h_src(num_rows * num_cols);
    std::vector<REAL> h_correct(num_rows * num_cols);
    std::iota(h_src.begin(), h_src.end(), 1.0);
    lambda_transpose(num_rows, num_cols, h_src, h_correct);
    BufferDevice<REAL> d_src(sycl_target, h_src);
    BufferDevice<REAL> d_dst(sycl_target, h_src);
    matrix_transpose(sycl_target, num_rows, num_cols, d_src.ptr, d_dst.ptr)
        .wait_and_throw();
    auto h_to_test = d_dst.get();
    ASSERT_EQ(h_correct, h_to_test);
  }
}

TEST(SYCLTarget, compare_and_swap_REAL) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t N = 1024000;
  std::vector<REAL> h_y(1);
  h_y.at(0) = N;

  BufferDevice d_y(sycl_target, h_y);

  auto k_y = d_y.ptr;
  sycl_target->queue
      .parallel_for<>(
          sycl_target->device_limits.validate_range_global(sycl::range<1>(N)),
          [=](sycl::id<1> idx) {
            const REAL value =
                static_cast<REAL>(idx) - static_cast<REAL>(N) / 2;
            atomic_fetch_min_cas_strong(k_y, value);
          })
      .wait_and_throw();

  d_y.get(h_y);
  ASSERT_EQ(h_y.at(0), -static_cast<REAL>(N) / 2);

  sycl_target->free();
}

TEST(SYCLTarget, compare_and_swap_INT) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  const std::size_t N = 1024000;

  std::vector<INT> h_x(N);

  std::mt19937 rng(52234234);
  std::uniform_int_distribution<INT> dist{std::numeric_limits<INT>::lowest(),
                                          std::numeric_limits<INT>::max()};

  INT correct = std::numeric_limits<INT>::max();
  for (std::size_t ix = 0; ix < N; ix++) {
    const INT v = dist(rng);
    h_x.at(ix) = v;
    correct = std::min(correct, v);
  }

  std::vector<INT> h_y(1);
  h_y.at(0) = N;

  BufferDevice d_x(sycl_target, h_x);
  BufferDevice d_y(sycl_target, h_y);

  auto k_x = d_x.ptr;
  auto k_y = d_y.ptr;

  sycl_target->queue
      .parallel_for<>(
          sycl_target->device_limits.validate_range_global(sycl::range<1>(N)),
          [=](sycl::id<1> idx) {
            const INT value = k_x[idx];
            atomic_fetch_min_cas_strong(k_y, value);
          })
      .wait_and_throw();

  d_y.get(h_y);
  ASSERT_EQ(h_y.at(0), correct);

  sycl_target->free();
}

TEST(SYCLTarget, atomics) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  ASSERT_TRUE(sycl_target->device_limits.check_atomics_sanity(
      sycl_target->queue, true));
  sycl_target->free();
}

TEST(SYCLTarget, atomics_long) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);
  auto queue = sycl_target->queue;

  ASSERT_TRUE(atomic_binop_check_long(queue, CheckAdd<int>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckAdd<INT>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckAdd<REAL>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckMin<int>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckMin<INT>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckMin<REAL>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckMax<int>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckMax<INT>{}));
  ASSERT_TRUE(atomic_binop_check_long(queue, CheckMax<REAL>{}));

  sycl_target->free();
}

TEST(SYCLTarget, buffer_max_int) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  std::random_device rng{};
  std::uniform_int_distribution<int> dist(-50, 50);

  BufferDevice<int> d_output(sycl_target, 1);

  for (std::size_t N : {1, 2, 31, 123, 1037}) {
    std::vector<int> h_input(N);

    int correct = std::numeric_limits<int>::lowest();
    for (auto &ix : h_input) {
      const int v = dist(rng);
      correct = std::max(correct, v);
      ix = v;
    }

    BufferDevice<int> d_input(sycl_target, h_input);

    reduce_values(sycl_target, N, d_input.ptr, sycl::maximum<int>(),
                  d_output.ptr)
        .wait_and_throw();

    auto h_output = d_output.get();

    ASSERT_EQ(h_output.at(0), correct);
  }

  sycl_target->free();
}

TEST(SYCLTarget, profile_region) {
  auto sycl_target = std::make_shared<SYCLTarget>(0, MPI_COMM_WORLD);

  sycl_target->profile_map.enable();

  constexpr int level = 1024;
  const std::string A = "A";
  auto r0 = sycl_target->profile_map.start_region(A, "B", level);

  sycl_target->profile_map.end_region(r0);

  ASSERT_EQ(sycl_target->profile_map.regions.size(), 1);
  ASSERT_EQ(sycl_target->profile_map.regions.front().key1, "A");
  ASSERT_EQ(sycl_target->profile_map.regions.front().key2, "B");
  ASSERT_EQ(sycl_target->profile_map.regions.front().level, level);

  sycl_target->profile_map.disable();
  auto r1 = sycl_target->profile_map.start_region("C", "D");

  sycl_target->profile_map.end_region(r1);

  ASSERT_EQ(sycl_target->profile_map.regions.size(), 1);
  ASSERT_EQ(sycl_target->profile_map.regions.front().key1, "A");
  ASSERT_EQ(sycl_target->profile_map.regions.front().key2, "B");
  ASSERT_EQ(sycl_target->profile_map.regions.front().level, level);

  sycl_target->profile_map.reset();

  ASSERT_EQ(sycl_target->profile_map.regions.size(), 0);
  sycl_target->free();
}
