#ifndef _NESO_PARTICLES_ALGORITHMS_CELLWISE_METHODS_HPP_
#define _NESO_PARTICLES_ALGORITHMS_CELLWISE_METHODS_HPP_

#include "../compute_target.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

/**
 * Get the number of particles in each cell as a CellDatConst.
 *
 * @param particle_sub_group Particle{Sub}Group containing particles.
 * @param cell_dat_const CellDatConst to populate with particle counts.
 * @param row Row to populate in the CellDatConst.
 * @param col Column to populate in the CellDatConst.
 */
template <typename GROUP_TYPE, typename VALUE_TYPE>
void get_npart_cell(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                    CellDatConstSharedPtr<VALUE_TYPE> cell_dat_const,
                    const int row = 0, const int col = 0) {

  auto particle_group = get_particle_group(particle_sub_group);
  const int cell_count = particle_group->domain->mesh->get_cell_count();

  NESOASSERT(cell_dat_const->ncells == cell_count,
             "Bad cell count in cell_dat_const");
  NESOASSERT(cell_dat_const->nrow > row, "Bad row passed: out of bounds.");
  NESOASSERT(cell_dat_const->ncol > col, "Bad col passed: out of bounds.");

  const int stride = cell_dat_const->nrow * cell_dat_const->ncol;
  const int nrow = cell_dat_const->nrow;
  auto sycl_target = particle_group->sycl_target;

  auto d_cell_dat_const_ptr = cell_dat_const->device_ptr();

  if constexpr (std::is_same<GROUP_TYPE, ParticleSubGroup>::value == true) {
    particle_sub_group->create_if_required();
    auto selection = particle_sub_group->get_selection();
    auto *k_npart_cell = selection.d_npart_cell;
    sycl_target->queue
        .parallel_for(sycl::range<1>(cell_count),
                      [=](auto idx) {
                        d_cell_dat_const_ptr[idx * stride + nrow * col + row] =
                            k_npart_cell[idx];
                      })
        .wait_and_throw();
  } else {
    auto *k_npart_cell = particle_group->cell_id_dat->d_npart_cell;
    sycl_target->queue
        .parallel_for(sycl::range<1>(cell_count),
                      [=](auto idx) {
                        d_cell_dat_const_ptr[idx * stride + nrow * col + row] =
                            k_npart_cell[idx];
                      })
        .wait_and_throw();
  }
}

extern template void get_npart_cell(std::shared_ptr<ParticleGroup>,
                                    CellDatConstSharedPtr<int>, int, int);
extern template void get_npart_cell(std::shared_ptr<ParticleGroup>,
                                    CellDatConstSharedPtr<INT>, int, int);
extern template void get_npart_cell(std::shared_ptr<ParticleSubGroup>,
                                    CellDatConstSharedPtr<int>, int, int);
extern template void get_npart_cell(std::shared_ptr<ParticleSubGroup>,
                                    CellDatConstSharedPtr<INT>, int, int);

namespace Private {

namespace CellDatConstLoop {

template <typename T> struct cell_dat_const_element_ptr_type;

template <typename T>
struct cell_dat_const_element_ptr_type<CellDatConstSharedPtr<T>> {
  using type = T *;
};

template <typename... DAT_ARGS>
struct cell_dat_const_loop_element_wise_loop_type {
  using type =
      Tuple::Tuple<typename cell_dat_const_element_ptr_type<DAT_ARGS>::type...>;
};

template <int INDEX, int SIZE, typename TUPLE_TYPE, typename ARG,
          typename... DAT_ARGS>
auto get_device_pointers_inner(TUPLE_TYPE &pointers, ARG &arg,
                               DAT_ARGS... dat_args) {
  Tuple::get<INDEX>(pointers) = arg->device_ptr();
  if constexpr ((INDEX + 1) < SIZE) {
    get_device_pointers_inner<INDEX + 1, SIZE>(pointers, dat_args...);
  }
}

template <typename... DAT_ARGS> auto get_device_pointers(DAT_ARGS... dat_args) {
  typename cell_dat_const_loop_element_wise_loop_type<DAT_ARGS...>::type
      pointers;
  get_device_pointers_inner<0, sizeof...(DAT_ARGS)>(pointers, dat_args...);
  return pointers;
}

template <typename T> struct cell_dat_const_element_type;

template <typename T>
struct cell_dat_const_element_type<CellDatConstSharedPtr<T>> {
  using type = T;
};

template <typename... DAT_ARGS>
struct cell_dat_const_loop_element_wise_kernel_type {
  using type =
      Tuple::Tuple<typename cell_dat_const_element_type<DAT_ARGS>::type...>;
};

template <int INDEX, int SIZE, typename LOOP_ARGS, typename KERNEL_ARGS>
void get_element_values(const std::size_t linear_index, LOOP_ARGS &loop_args,
                        const std::size_t index_cell,
                        const std::size_t index_row,
                        const std::size_t index_col, KERNEL_ARGS &kernel_args) {

  Tuple::get<INDEX>(kernel_args) = Tuple::get<INDEX>(loop_args)[linear_index];
  if constexpr ((INDEX + 1) < SIZE) {
    get_element_values<INDEX + 1, SIZE>(linear_index, loop_args, index_cell,
                                        index_row, index_col, kernel_args);
  }
}

} // namespace CellDatConstLoop
} // namespace Private

/*
 * Applies a kernel element wise to the input CellDatConst dats and assigns the
 * output to the result dat.
 *
 * // For CellDatConstSharedPtrs a,b,c and d.
 * cell_dat_const_loop_element_wise(
 *     d,
 *     // This kernel is executed element wise.
 *     [=](auto a, auto b, auto c){
 *         return a * b + c;
 *     },
 *     a, b, c
 * );
 *
 * @param result_dat CellDatConst to be overwritten with the result of the
 * operation. May be equal to one of the arguments to the kernel.
 * @param kernel Kernel to apply element wise to compute the result. The
 * parameters of the kernel should be scalar types correpsonding to each of the
 * remaining arguments of this function.
 * @param dat_args CellDatConstSharedPtrs which provide the elements to pass to
 * the kernel.
 */
template <typename T, typename KERNEL_TYPE, typename... DAT_ARGS>
inline void
cell_dat_const_loop_element_wise(CellDatConstSharedPtr<T> result_dat,
                                 KERNEL_TYPE kernel, DAT_ARGS... dat_args) {
  auto sycl_target = result_dat->sycl_target;
  auto r0 = sycl_target->profile_map.start_region(
      "cell_dat_const_loop_element_wise", typeid(kernel).name());
  const int cell_count = result_dat->ncells;
  const int nrow = result_dat->nrow;
  const int ncol = result_dat->ncol;

  auto lambda_check_args = [&](auto dat) {
    NESOASSERT(dat->sycl_target == sycl_target, "Missmatched SYCLTarget.");
    NESOASSERT(dat->ncells == cell_count, "Missmatched cell count.");
    NESOASSERT(dat->nrow == nrow, "Missmatched number of rows.");
    NESOASSERT(dat->ncol == ncol, "Missmatched number of columns.");
  };
  (lambda_check_args(dat_args), ...);

  const auto pointers =
      Private::CellDatConstLoop::get_device_pointers(dat_args...);

  auto *output_pointer = result_dat->device_ptr();

  sycl_target->queue
      .parallel_for(
          sycl_target->device_limits.validate_range_global(
              sycl::range<3>(cell_count, ncol, nrow)),
          [=](sycl::item<3> idx) {
            const std::size_t index_cell = idx.get_id(0);
            const std::size_t index_col = idx.get_id(1);
            const std::size_t index_row = idx.get_id(2);

            const std::size_t linear_index =
                nrow * index_col + index_row + index_cell * nrow * ncol;

            typename Private::CellDatConstLoop::
                cell_dat_const_loop_element_wise_kernel_type<DAT_ARGS...>::type
                    kernel_args;

            Private::CellDatConstLoop::get_element_values<0,
                                                          sizeof...(DAT_ARGS)>(
                linear_index, pointers, index_cell, index_row, index_col,
                kernel_args);

            output_pointer[linear_index] =
                static_cast<T>(Tuple::apply(kernel, kernel_args));
          }

          )
      .wait_and_throw();
  sycl_target->profile_map.end_region(r0);
}

} // namespace NESO::Particles

#endif
