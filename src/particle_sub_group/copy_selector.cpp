#include <neso_particles/particle_sub_group/copy_selector.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group.hpp>
#include <numeric>

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

CopySelector::CopySelector(std::shared_ptr<ParticleSubGroup> parent)
    : SubGroupSelectorBase(parent), parent(parent) {
  this->add_parent_dependencies(parent);
  this->is_whole_particle_group = parent->is_entire_particle_group();
}

void CopySelector::create(Selection *created_selection) {

  if (this->is_whole_particle_group) {
    this->parent->create_if_required();
    auto parent_selection = this->parent->get_selection();
    *created_selection = parent_selection;
  } else {
    auto sycl_target = this->particle_group->sycl_target;
    this->parent->create_if_required();
    auto parent_selection = this->parent->get_selection();

    auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
          d_npart_cell_es_ptr] =
        this->sub_group_particle_map->get_helper_ptrs();

    const auto cell_count = parent_selection.ncell;

    auto k_npart_cell_ptr = d_npart_cell_ptr;
    auto k_npart_cell_es_ptr = d_npart_cell_es_ptr;
    auto k_parent_npart_cell_ptr = parent_selection.d_npart_cell;
    auto k_parent_npart_cell_es_ptr = parent_selection.d_npart_cell_es;

    auto e0 = sycl_target->queue.parallel_for(
        sycl::range<1>(cell_count), [=](auto ix) {
          k_npart_cell_ptr[ix] = k_parent_npart_cell_ptr[ix];
          k_npart_cell_es_ptr[ix] = k_parent_npart_cell_es_ptr[ix];
        });
    std::memcpy(h_npart_cell_ptr, parent_selection.h_npart_cell,
                cell_count * sizeof(int));
    std::exclusive_scan(h_npart_cell_ptr, h_npart_cell_ptr + cell_count,
                        h_npart_cell_es_ptr, static_cast<INT>(0));

    this->sub_group_particle_map->create(0, cell_count, h_npart_cell_ptr,
                                         h_npart_cell_es_ptr);
    auto k_cell_starts_ptr = this->sub_group_particle_map->d_cell_starts->ptr;
    auto k_cell_starts_parent_ptr =
        parent_selection.d_map_cells_to_particles.map_ptr;

    const std::size_t local_size =
        sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;

    e0.wait_and_throw();
    auto e1 = sycl_target->queue.parallel_for(
        sycl_target->device_limits.validate_nd_range(sycl::nd_range<2>(
            sycl::range<2>(static_cast<std::size_t>(cell_count), local_size),
            sycl::range<2>(1, local_size))),
        [=](auto idx) {
          const std::size_t cell = idx.get_global_id(0);
          const std::size_t local_id = idx.get_local_id(1);
          const auto npart_cell = k_npart_cell_ptr[cell];
          std::size_t layer = local_id;
          while (layer < npart_cell) {
            k_cell_starts_ptr[cell][layer] =
                k_cell_starts_parent_ptr[cell][layer];
            layer += local_size;
          }
        });

    created_selection->npart_local = parent_selection.npart_local;
    created_selection->ncell = cell_count;
    created_selection->h_npart_cell = h_npart_cell_ptr;
    created_selection->d_npart_cell = d_npart_cell_ptr;
    created_selection->d_npart_cell_es = d_npart_cell_es_ptr;
    created_selection->d_map_cells_to_particles = {k_cell_starts_ptr};
    e1.wait_and_throw();
  }
}

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles
