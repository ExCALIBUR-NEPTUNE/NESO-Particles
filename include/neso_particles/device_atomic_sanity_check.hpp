#ifndef _NESO_PARTICLES_DEVICE_ATOMIC_SANITY_CHECK_HPP_
#define _NESO_PARTICLES_DEVICE_ATOMIC_SANITY_CHECK_HPP_

/**
 * This file contains the implementation to test if a SYCL implementation has
 * fundamentally broken atomics.
 */

#include "device_functions.hpp"
#include <random>
#include <vector>

namespace NESO::Particles {

template <typename T> struct CheckAdd {
  static inline T binop_host(T *ptr, const T value) {
    const T current = *ptr;
    *ptr = current + value;
    return current;
  }
  static inline T binop_device(T *ptr, const T value) {
    return atomic_fetch_add(ptr, value);
  }
  static constexpr T identity_element = 0;
  static inline bool test(const T correct, const T to_test) {
    const T err_abs = std::abs(correct - to_test);
    const T scaling = std::abs(correct);
    const T err_rel = (scaling > 0) ? err_abs / scaling : err_abs;
    return (err_rel < 1e-8) || (err_abs < 1e-8);
  }
};

template <typename T> struct CheckMin {
  static inline T binop_host(T *ptr, const T value) {
    const T current = *ptr;
    *ptr = std::min(current, value);
    return current;
  }
  static inline T binop_device(T *ptr, const T value) {
    return atomic_fetch_min(ptr, value);
  }
  static constexpr T identity_element = std::numeric_limits<T>::max();
  static inline bool test(const T correct, const T to_test) {
    return correct == to_test;
  }
};

template <typename T> struct CheckMax {
  static inline T binop_host(T *ptr, const T value) {
    const T current = *ptr;
    *ptr = std::max(current, value);
    return current;
  }
  static inline T binop_device(T *ptr, const T value) {
    return atomic_fetch_max(ptr, value);
  }
  static constexpr T identity_element = std::numeric_limits<T>::lowest();
  static inline bool test(const T correct, const T to_test) {
    return correct == to_test;
  }
};

template <template <typename> typename SPEC, typename T>
inline bool atomic_binop_check(sycl::queue queue, const SPEC<T> spec) {
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

  auto e0 =
      queue.parallel_for<>(sycl::range<1>(num_elements), [=](sycl::id<1> idx) {
        spec.binop_device(d_result, d_ptr[idx]);
      });

  for (auto &ex : elements) {
    spec.binop_host(&h_correct, ex);
  }

  e0.wait_and_throw();
  queue.memcpy(&h_to_test, d_result, sizeof(T)).wait_and_throw();

  const bool passed = spec.test(h_correct, h_to_test);

  sycl::free(d_ptr, queue);
  sycl::free(d_result, queue);
  return passed;
}

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

} // namespace NESO::Particles

#endif
