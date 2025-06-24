#ifndef _NESO_PARTICLES_ALGORITHMS_REDUCE_DAT_CELLWISE_HPP_
#define _NESO_PARTICLES_ALGORITHMS_REDUCE_DAT_CELLWISE_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat_const.hpp"
#include "../device_functions.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

namespace Private {

template <typename T>
inline ParticleDatImplGetConstT<T>
particle_dat_impl_get_const(ParticleDatSharedPtr<T> dat) {
  return dat->impl_get_const();
}

template <typename T>
inline CellDatConstDeviceType<T>
cell_dat_const_impl_get(CellDatConstSharedPtr<T> cell_dat_const) {
  return cell_dat_const->impl_get();
}

template <typename T, typename OP>
inline sycl::event reduce_dat_component_cellwise_async(
    ParticleGroupSharedPtr particle_group, Sym<T> sym, const int sym_component,
    CellDatConstSharedPtr<T> cell_dat_const, const int cell_dat_const_row,
    const int cell_dat_const_col, OP op) {
  auto sycl_target = particle_group->sycl_target;
  auto dat = particle_group->get_dat(sym);

  const std::size_t cell_count =
      static_cast<std::size_t>(particle_group->domain->mesh->get_cell_count());
  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;
  sycl::range<2> iterset_outer{cell_count, local_size};
  sycl::range<2> iterset_inner{1, local_size};

  const auto d_dat_ptr = particle_dat_impl_get_const(dat);
  auto d_cell_dat_const = cell_dat_const_impl_get(cell_dat_const);
  auto d_npart_cell = dat->d_npart_cell;

  sycl::event event = sycl_target->queue.parallel_for(
      sycl_target->device_limits.validate_nd_range(
          sycl::nd_range<2>(iterset_outer, iterset_inner)),
      [=](sycl::nd_item<2> idx) {
        const std::size_t cellx = idx.get_global_id(0);
        const auto npart_cell = d_npart_cell[cellx];
        if (npart_cell > 0) {
          auto d_start = d_dat_ptr[cellx][sym_component];
          auto d_end = d_start + npart_cell;
          std::size_t workitem_id = idx.get_local_id(1);
          T value = Kernel::joint_reduce(idx.get_group(), d_start, d_end, op);

          if (workitem_id == 0) {
            auto ptr = d_cell_dat_const.ptr;
            auto stride = d_cell_dat_const.stride;
            auto nrow = d_cell_dat_const.nrow;
            const std::size_t index = static_cast<std::size_t>(
                cellx * stride + nrow * cell_dat_const_col +
                cell_dat_const_row);
            const T current = ptr[index];
            ptr[index] = op(current, value);
          }
        }
      });

  return event;
}

template <typename T, typename OP>
inline sycl::event reduce_dat_component_cellwise_async(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<T> sym,
    const int sym_component, CellDatConstSharedPtr<T> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col, OP op) {
  auto particle_group = get_particle_group(particle_sub_group);

  if (particle_sub_group->is_entire_particle_group()) {
    return reduce_dat_component_cellwise_async(
        particle_group, sym, sym_component, cell_dat_const, cell_dat_const_row,
        cell_dat_const_col, op);
  } else {

    auto sycl_target = particle_group->sycl_target;
    auto dat = particle_group->get_dat(sym);

    const std::size_t cell_count = static_cast<std::size_t>(
        particle_group->domain->mesh->get_cell_count());
    const std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    sycl::range<2> iterset_outer{cell_count, local_size};
    sycl::range<2> iterset_inner{1, local_size};

    const auto d_dat_ptr = particle_dat_impl_get_const(dat);
    auto d_cell_dat_const = cell_dat_const_impl_get(cell_dat_const);

    particle_sub_group->create_if_required();
    auto selection = particle_sub_group->get_selection();
    auto d_npart_cell = selection.d_npart_cell;
    auto d_map = selection.d_map_cells_to_particles;

    sycl::event event = sycl_target->queue.parallel_for(
        sycl_target->device_limits.validate_nd_range(
            sycl::nd_range<2>(iterset_outer, iterset_inner)),
        [=](sycl::nd_item<2> idx) {
          const std::size_t cellx = idx.get_global_id(0);
          const auto npart_cell = d_npart_cell[cellx];
          if (npart_cell > 0) {
            auto d_ptr = d_dat_ptr[cellx][sym_component];
            std::size_t workitem_id = idx.get_local_id(1);
            T value = Kernel::get_identity(op);

            const std::size_t stride = idx.get_local_range(1);
            for (std::size_t index = workitem_id;
                 index < static_cast<std::size_t>(npart_cell);
                 index += stride) {
              const INT layer = d_map.map_loop_layer_to_layer(cellx, index);
              value = op(value, d_ptr[layer]);
            }

            value = sycl::reduce_over_group(idx.get_group(), value, op);

            if (workitem_id == 0) {
              auto ptr = d_cell_dat_const.ptr;
              auto stride = d_cell_dat_const.stride;
              auto nrow = d_cell_dat_const.nrow;
              const std::size_t index = static_cast<std::size_t>(
                  cellx * stride + nrow * cell_dat_const_col +
                  cell_dat_const_row);
              const T current = ptr[index];
              ptr[index] = op(current, value);
            }
          }
        });

    return event;
  }
}

template <typename T, typename OP>
inline sycl::event reduce_dat_components_cellwise_async(
    ParticleGroupSharedPtr particle_group, Sym<T> sym,
    CellDatConstSharedPtr<T> cell_dat_const, int *d_output_offsets, OP op) {
  auto sycl_target = particle_group->sycl_target;
  auto dat = particle_group->get_dat(sym);
  const auto num_components = dat->ncomp;

  const std::size_t cell_count =
      static_cast<std::size_t>(particle_group->domain->mesh->get_cell_count());
  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;
  sycl::range<2> iterset_outer{cell_count * num_components, local_size};
  sycl::range<2> iterset_inner{1, local_size};

  const auto d_dat_ptr = particle_dat_impl_get_const(dat);
  auto d_cell_dat_const = cell_dat_const_impl_get(cell_dat_const);
  auto d_npart_cell = dat->d_npart_cell;

  sycl::event event = sycl_target->queue.parallel_for(
      sycl_target->device_limits.validate_nd_range(
          sycl::nd_range<2>(iterset_outer, iterset_inner)),
      [=](sycl::nd_item<2> idx) {
        const std::size_t cellx_compx = idx.get_global_id(0);
        const std::size_t cellx =
            cellx_compx / static_cast<std::size_t>(num_components);
        const std::size_t sym_component = cellx_compx - cellx * num_components;
        const auto npart_cell = d_npart_cell[cellx];

        if (npart_cell > 0) {
          auto d_start = d_dat_ptr[cellx][sym_component];
          auto d_end = d_start + npart_cell;
          std::size_t workitem_id = idx.get_local_id(1);
          T value = Kernel::joint_reduce(idx.get_group(), d_start, d_end, op);

          if (workitem_id == 0) {
            auto ptr = d_cell_dat_const.ptr;
            auto stride = d_cell_dat_const.stride;
            const std::size_t index = static_cast<std::size_t>(
                cellx * stride + d_output_offsets[sym_component]);
            const T current = ptr[index];
            ptr[index] = op(current, value);
          }
        }
      });

  return event;
}

template <typename T, typename OP>
inline sycl::event reduce_dat_components_cellwise_async(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<T> sym,
    CellDatConstSharedPtr<T> cell_dat_const, int *d_output_offsets, OP op) {

  auto particle_group = get_particle_group(particle_sub_group);
  if (particle_sub_group->is_entire_particle_group()) {
    return reduce_dat_components_cellwise_async(
        particle_group, sym, cell_dat_const, d_output_offsets, op);
  } else {

    auto sycl_target = particle_group->sycl_target;
    auto dat = particle_group->get_dat(sym);
    const auto num_components = dat->ncomp;

    const std::size_t cell_count = static_cast<std::size_t>(
        particle_group->domain->mesh->get_cell_count());
    const std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;
    sycl::range<2> iterset_outer{cell_count * num_components, local_size};
    sycl::range<2> iterset_inner{1, local_size};

    const auto d_dat_ptr = particle_dat_impl_get_const(dat);
    auto d_cell_dat_const = cell_dat_const_impl_get(cell_dat_const);

    particle_sub_group->create_if_required();
    auto selection = particle_sub_group->get_selection();
    auto d_npart_cell = selection.d_npart_cell;
    auto d_map = selection.d_map_cells_to_particles;

    sycl::event event = sycl_target->queue.parallel_for(
        sycl_target->device_limits.validate_nd_range(
            sycl::nd_range<2>(iterset_outer, iterset_inner)),
        [=](sycl::nd_item<2> idx) {
          const std::size_t cellx_compx = idx.get_global_id(0);
          const std::size_t cellx =
              cellx_compx / static_cast<std::size_t>(num_components);
          const std::size_t sym_component =
              cellx_compx - cellx * num_components;
          const auto npart_cell = d_npart_cell[cellx];
          if (npart_cell > 0) {
            auto d_ptr = d_dat_ptr[cellx][sym_component];
            std::size_t workitem_id = idx.get_local_id(1);
            T value = Kernel::get_identity(op);

            const std::size_t stride = idx.get_local_range(1);
            for (std::size_t index = workitem_id; index < npart_cell;
                 index += stride) {
              const INT layer = d_map.map_loop_layer_to_layer(cellx, index);
              value = op(value, d_ptr[layer]);
            }

            value = sycl::reduce_over_group(idx.get_group(), value, op);

            if (workitem_id == 0) {
              auto ptr = d_cell_dat_const.ptr;
              auto stride = d_cell_dat_const.stride;
              const std::size_t index = static_cast<std::size_t>(
                  cellx * stride + d_output_offsets[sym_component]);
              const T current = ptr[index];
              ptr[index] = op(current, value);
            }
          }
        });

    return event;
  }
}

} // namespace Private

/**
 * Reduces a single value from each particle into a specified CellDatConst
 * index.
 *
 * @param[in] particle_sub_group ParticleGroup or ParticleSubGroup providing
 * source particles.
 * @param[in] sym Specify the source particle property.
 * @param[in] sym_component Specify the index in the particle property to
 * reduce.
 * @param[in, out] cell_dat_const The output CellDatConst to reduce into. The
 * reduced values from the particles will be combined with the value already in
 * the CellDatConst.
 * @param[in] cell_dat_const_row The row to reduce into of the CellDatConst.
 * @param[in] cell_dat_const_cel The column to reduce into of the CellDatConst.
 * @param[in] op Binary operation to apply.
 */
template <typename GROUP_TYPE, typename T, typename OP>
void reduce_dat_component_cellwise(
    std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<T> sym,
    const int sym_component, CellDatConstSharedPtr<T> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col, OP op) {

  auto particle_group = get_particle_group(particle_sub_group);
  const std::size_t cell_count =
      static_cast<std::size_t>(particle_group->domain->mesh->get_cell_count());

  NESOASSERT(cell_count == static_cast<std::size_t>(cell_dat_const->ncells),
             "Missmatch in cell count.");
  NESOASSERT(particle_group->contains_dat(sym),
             "Sym not in Particle{Sub}Group.");
  NESOASSERT((0 <= cell_dat_const_row) &&
                 (cell_dat_const_row < cell_dat_const->nrow),
             "The passed cell_dat_const_row is incompatible with the "
             "CellDatConst passed.");
  NESOASSERT((0 <= cell_dat_const_col) &&
                 (cell_dat_const_col < cell_dat_const->ncol),
             "The passed cell_dat_const_row is incompatible with the "
             "CellDatConst passed.");

  Private::reduce_dat_component_cellwise_async(
      particle_sub_group, sym, sym_component, cell_dat_const,
      cell_dat_const_row, cell_dat_const_col, op)
      .wait_and_throw();
}

extern template void reduce_dat_component_cellwise(
    ParticleGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int sym_component, CellDatConstSharedPtr<REAL> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<REAL> op);

extern template void reduce_dat_component_cellwise(
    ParticleGroupSharedPtr particle_sub_group, Sym<INT> sym,
    const int sym_component, CellDatConstSharedPtr<INT> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<INT> op);

extern template void reduce_dat_component_cellwise(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int sym_component, CellDatConstSharedPtr<REAL> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<REAL> op);

extern template void reduce_dat_component_cellwise(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<INT> sym,
    const int sym_component, CellDatConstSharedPtr<INT> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<INT> op);

/**
 * Reduces all particle properties for a given property into a CellDatConst
 * cellwise. CellDatConst elements are indexed row-wise. The total number of
 * CellDatConst elements must match the number of components of the particle
 * property.
 *
 * @param[in] particle_sub_group ParticleGroup or ParticleSubGroup providing
 * source particles.
 * @param[in] sym Specify the source particle property.
 * @param[in, out] cell_dat_const The output CellDatConst to reduce into. The
 * reduced values from the particles will be combined with the value already in
 * the CellDatConst.
 * @param[in] op Binary operation to apply.
 */
template <typename GROUP_TYPE, typename T, typename OP>
void reduce_dat_components_cellwise(
    std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<T> sym,
    CellDatConstSharedPtr<T> cell_dat_const, OP op) {

  auto particle_group = get_particle_group(particle_sub_group);

  const std::size_t cell_count =
      static_cast<std::size_t>(particle_group->domain->mesh->get_cell_count());

  NESOASSERT(cell_count == static_cast<std::size_t>(cell_dat_const->ncells),
             "Missmatch in cell count.");
  NESOASSERT(particle_group->contains_dat(sym),
             "Sym not in Particle{Sub}Group.");

  auto dat_ncomp = particle_group->get_dat(sym)->ncomp;

  const auto nrow = cell_dat_const->nrow;
  const auto ncol = cell_dat_const->ncol;
  auto cell_dat_const_nelements = nrow * ncol;
  NESOASSERT(dat_ncomp >= cell_dat_const_nelements,
             "Missmatch between the number of particle components: " +
                 std::to_string(dat_ncomp) +
                 " and the matrix size of the CellDatConst: " +
                 std::to_string(cell_dat_const_nelements));

  auto sycl_target = particle_group->sycl_target;
  auto d_buffer =
      get_resource<BufferDevice<int>, ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
  d_buffer->realloc_no_copy(dat_ncomp);

  std::vector<int> h_buffer(dat_ncomp);

  int index = 0;
  for (int rx = 0; rx < nrow; rx++) {
    for (int cx = 0; cx < ncol; cx++) {
      h_buffer.at(index) = nrow * cx + rx;
      index++;
    }
  }

  sycl_target->queue
      .memcpy(d_buffer->ptr, h_buffer.data(), dat_ncomp * sizeof(int))
      .wait_and_throw();

  Private::reduce_dat_components_cellwise_async(
      particle_sub_group, sym, cell_dat_const, d_buffer->ptr, op)
      .wait_and_throw();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<int>{}, d_buffer);
}

extern template void reduce_dat_components_cellwise(
    ParticleGroupSharedPtr particle_group, Sym<REAL> sym,
    CellDatConstSharedPtr<REAL> cell_dat_const, Kernel::plus<REAL> op);

extern template void reduce_dat_components_cellwise(
    ParticleGroupSharedPtr particle_group, Sym<INT> sym,
    CellDatConstSharedPtr<INT> cell_dat_const, Kernel::plus<INT> op);

extern template void reduce_dat_components_cellwise(
    ParticleSubGroupSharedPtr particle_group, Sym<REAL> sym,
    CellDatConstSharedPtr<REAL> cell_dat_const, Kernel::plus<REAL> op);

extern template void reduce_dat_components_cellwise(
    ParticleSubGroupSharedPtr particle_group, Sym<INT> sym,
    CellDatConstSharedPtr<INT> cell_dat_const, Kernel::plus<INT> op);

} // namespace NESO::Particles

#endif
