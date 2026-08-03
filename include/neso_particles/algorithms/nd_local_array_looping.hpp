#ifndef _NESO_PARTICLES_ALGORITHMS_ND_LOCAL_ARRAY_LOOPING_HPP_
#define _NESO_PARTICLES_ALGORITHMS_ND_LOCAL_ARRAY_LOOPING_HPP_

#include "../containers/nd_local_array.hpp"

namespace NESO::Particles {

namespace Private {
namespace NDLocalArrayLoop {

template <typename T> struct nd_local_array_element_ptr_type;

template <typename T, std::size_t N>
struct nd_local_array_element_ptr_type<NDLocalArray<T, N>> {
  using type = T *;
};

template <typename T, std::size_t N>
struct nd_local_array_element_ptr_type<NDLocalArraySharedPtr<T, N>> {
  using type = T *;
};

template <typename... ARRAY_ARGS>
struct nd_local_array_loop_element_wise_loop_type {
  using type = Tuple::Tuple<
      typename nd_local_array_element_ptr_type<ARRAY_ARGS>::type...>;
};

template <int INDEX, int SIZE, typename TUPLE_TYPE, typename ARG,
          typename... ARRAY_ARGS>
auto get_device_pointers_inner(TUPLE_TYPE &pointers, ARG &arg,
                               ARRAY_ARGS... array_args) {
  Tuple::get<INDEX>(pointers) = arg->ptr();
  if constexpr ((INDEX + 1) < SIZE) {
    get_device_pointers_inner<INDEX + 1, SIZE>(pointers, array_args...);
  }
}

template <typename... ARRAY_ARGS>
auto get_device_pointers(ARRAY_ARGS... array_args) {
  typename nd_local_array_loop_element_wise_loop_type<ARRAY_ARGS...>::type
      pointers;
  get_device_pointers_inner<0, sizeof...(ARRAY_ARGS)>(pointers, array_args...);
  return pointers;
}

template <typename T> struct nd_local_array_element_type;

template <typename T, std::size_t N>
struct nd_local_array_element_type<NDLocalArraySharedPtr<T, N>> {
  using type = T;
};

template <typename... ARRAY_ARGS>
struct nd_local_array_loop_element_wise_kernel_type {
  using type =
      Tuple::Tuple<typename nd_local_array_element_type<ARRAY_ARGS>::type...>;
};

template <int INDEX, int SIZE, typename LOOP_ARGS, typename KERNEL_ARGS>
void get_element_values(const std::size_t linear_index, LOOP_ARGS &loop_args,
                        KERNEL_ARGS &kernel_args) {

  Tuple::get<INDEX>(kernel_args) = Tuple::get<INDEX>(loop_args)[linear_index];
  if constexpr ((INDEX + 1) < SIZE) {
    get_element_values<INDEX + 1, SIZE>(linear_index, loop_args, kernel_args);
  }
}

template <int INDEX, int SIZE, typename ARRAY, typename... ARRAY_ARGS>
void get_broadcast_masks(std::array<int, SIZE> &masks, ARRAY &a,
                         ARRAY_ARGS... array_args) {

  const bool broadcast = a->index.size() == 1;
  masks[INDEX] = broadcast ? 1 : 0;

  if constexpr (INDEX + 1 < SIZE) {
    get_broadcast_masks<INDEX + 1, SIZE>(masks, array_args...);
  }
}

template <int INDEX, int SIZE, typename LOOP_ARGS, typename KERNEL_ARGS>
void get_element_values_broadcast(const std::size_t linear_index,
                                  LOOP_ARGS &loop_args,
                                  const std::array<int, SIZE> &masks,
                                  KERNEL_ARGS &kernel_args) {

  // This index becomes zero when the broadcast mask is set.
  const std::size_t index =
      linear_index * (1 - static_cast<std::size_t>(masks[INDEX]));

  Tuple::get<INDEX>(kernel_args) = Tuple::get<INDEX>(loop_args)[index];
  if constexpr ((INDEX + 1) < SIZE) {
    get_element_values_broadcast<INDEX + 1, SIZE>(linear_index, loop_args,
                                                  masks, kernel_args);
  }
}
} // namespace NDLocalArrayLoop
} // namespace Private

/*
 * Applies a kernel element wise to the input NDLocalArrays and assigns the
 * output to the result local array.
 *
 * // For NDLocalArraySharedPtrs a,b,c and d.
 * nd_local_array_loop_element_wise(
 *     d,
 *     // This kernel is executed element wise.
 *     [=](auto a, auto b, auto c){
 *         return a * b + c;
 *     },
 *     a, b, c
 * );
 *
 * If the output array, d, has shape NxM and if any arguments have nrow==1 and
 * ncol==1 then these single values are logically broadcast to size NxM.
 *
 * @param result_array NDLocalArray to be overwritten with the result of the
 * operation. May be equal to one of the arguments to the kernel.
 * @param kernel Kernel to apply element wise to compute the result. The
 * parameters of the kernel should be scalar types correpsonding to each of the
 * remaining arguments of this function.
 * @param array_args NDLocalArraySharedPtrs which provide the elements to pass
 * to the kernel.
 */
template <typename T, std::size_t N, typename KERNEL_TYPE,
          typename... ARRAY_ARGS>
inline void
nd_local_array_loop_element_wise(NDLocalArraySharedPtr<T, N> result_array,
                                 KERNEL_TYPE kernel, ARRAY_ARGS... array_args) {

  auto sycl_target = result_array->sycl_target;
  auto r0 = sycl_target->profile_map.start_region(
      "nd_local_array_loop_element_wise", typeid(kernel).name());

  NDIndex<N> shape = result_array->index;
  bool broadcasting = false;

  auto lambda_check_args = [&](auto a) {
    NESOASSERT(a->sycl_target == sycl_target, "Missmatched SYCLTarget.");
    if (a->index.size() == 1) {
      broadcasting = true;
    } else {
      NESOASSERT(a->index == shape,
                 "Missmatched shape between input and output arrays.");
    }
  };
  (lambda_check_args(array_args), ...);

  auto iteration_set = sycl_target->device_limits.validate_range_global(
      sycl::range<1>(shape.size()));

  auto *output_pointer = result_array->ptr();
  const auto pointers =
      Private::NDLocalArrayLoop::get_device_pointers(array_args...);

  if (broadcasting) {

    constexpr int num_arrays = sizeof...(ARRAY_ARGS);
    std::array<int, num_arrays> broadcast_masks = {0};
    Private::NDLocalArrayLoop::get_broadcast_masks<0, num_arrays>(
        broadcast_masks, array_args...);

    sycl_target->queue
        .parallel_for(iteration_set,
                      [=](sycl::item<1> ix) {
                        const std::size_t linear_index = ix.get_id();
                        typename Private::NDLocalArrayLoop::
                            nd_local_array_loop_element_wise_kernel_type<
                                ARRAY_ARGS...>::type kernel_args;

                        Private::NDLocalArrayLoop::get_element_values_broadcast<
                            0, sizeof...(ARRAY_ARGS)>(linear_index, pointers,
                                                      broadcast_masks,
                                                      kernel_args);

                        output_pointer[linear_index] =
                            static_cast<T>(Tuple::apply(kernel, kernel_args));
                      })
        .wait_and_throw();

  } else {

    sycl_target->queue
        .parallel_for(iteration_set,
                      [=](sycl::item<1> ix) {
                        const std::size_t linear_index = ix.get_id();
                        typename Private::NDLocalArrayLoop::
                            nd_local_array_loop_element_wise_kernel_type<
                                ARRAY_ARGS...>::type kernel_args;

                        Private::NDLocalArrayLoop::get_element_values<
                            0, sizeof...(ARRAY_ARGS)>(linear_index, pointers,
                                                      kernel_args);

                        output_pointer[linear_index] =
                            static_cast<T>(Tuple::apply(kernel, kernel_args));
                      })
        .wait_and_throw();
  }

  sycl_target->profile_map.end_region(r0);
}
} // namespace NESO::Particles

#endif
