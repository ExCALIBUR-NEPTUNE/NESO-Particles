#include <neso_particles/common_impl.hpp>
#include <neso_particles/particle_sub_group/cell_sub_group_selector.hpp>

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleGroup> parent, const int cell_start,
    const int cell_end);
template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleSubGroup> parent, const int cell_start,
    const int cell_end);

template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleGroup> parent, const int cell);
template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleSubGroup> parent, const int cell);

void CellSubGroupSelector::create(Selection *created_selection) {

  const int cell_count = this->particle_group->domain->mesh->get_cell_count();
  auto sycl_target = this->particle_group->sycl_target;

  auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
        d_npart_cell_es_ptr] = this->sub_group_particle_map->get_helper_ptrs();

  const auto npart_local = this->particle_group->get_npart_local();
  const int range_cell_count = this->cell_end - this->cell_start;

  std::fill(h_npart_cell_ptr, h_npart_cell_ptr + cell_count, 0);

  if (this->parent_is_whole_group) {
    auto pr0 = ProfileRegion("CellSubGroupSelector", "create_particle_group");

    INT es_tmp = 0;
    INT max_occ = 0;
    for (int cell = cell_start; cell < cell_end; cell++) {
      const INT total = this->particle_group->get_npart_cell(cell);
      max_occ = std::max(total, max_occ);
      h_npart_cell_ptr[cell] = total;
      h_npart_cell_es_ptr[cell] = es_tmp;
      es_tmp += total;
    }

    EventStack es;

    if (range_cell_count > 0) {
      es.push(sycl_target->queue.memcpy(d_npart_cell_es_ptr + cell_start,
                                        h_npart_cell_es_ptr + cell_start,
                                        sizeof(INT) * range_cell_count));
      es.push(sycl_target->queue.memcpy(d_npart_cell_ptr + cell_start,
                                        h_npart_cell_ptr + cell_start,
                                        range_cell_count * sizeof(int)));
    }

    this->sub_group_particle_map->create(cell_start, cell_end, h_npart_cell_ptr,
                                         h_npart_cell_es_ptr);

    if ((range_cell_count > 0) && (max_occ > 0)) {
      const std::size_t local_size =
          sycl_target->parameters
              ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
              ->value;
      const std::size_t range_cell = get_next_multiple(max_occ, local_size);

      const auto k_npart_cell = d_npart_cell_ptr;
      const auto k_cell_starts =
          this->sub_group_particle_map->d_cell_starts->ptr;
      const auto k_cell_start = this->cell_start;

      es.wait();
      es.push(sycl_target->queue.parallel_for(
          sycl_target->device_limits.validate_nd_range(
              sycl::nd_range<2>(sycl::range<2>(range_cell_count, range_cell),
                                sycl::range<2>(1, local_size))),
          [=](sycl::nd_item<2> idx) {
            const std::size_t index_cell = idx.get_global_id(0) + k_cell_start;
            const std::size_t index_layer = idx.get_global_id(1);
            if (index_layer < k_npart_cell[index_cell]) {
              k_cell_starts[index_cell][index_layer] = index_layer;
            }
          }));
    }

    es.wait();

    Selection s;
    s.npart_local = es_tmp;
    s.ncell = cell_count;
    s.h_npart_cell = h_npart_cell_ptr;
    s.d_npart_cell = d_npart_cell_ptr;
    s.d_npart_cell_es = d_npart_cell_es_ptr;
    s.d_map_cells_to_particles = {
        this->sub_group_particle_map->d_cell_starts->ptr};
    *created_selection = s;

    pr0.end();
    sycl_target->profile_map.add_region(pr0);
  } else {
    auto pr0 =
        ProfileRegion("CellSubGroupSelector", "create_particle_sub_group");

    NESOASSERT(this->particle_sub_group != nullptr,
               "Parent is a sub group but we have no ParticleSubGroup.");

    this->particle_sub_group->create_if_required();
    auto s_parent = this->particle_sub_group->get_selection();

    INT es_tmp = 0;
    INT max_occ = 0;
    for (int cell = cell_start; cell < cell_end; cell++) {
      const INT total = s_parent.h_npart_cell[cell];
      max_occ = std::max(total, max_occ);
      h_npart_cell_ptr[cell] = total;
      h_npart_cell_es_ptr[cell] = es_tmp;
      es_tmp += total;
    }

    EventStack es;
    if (range_cell_count > 0) {
      es.push(sycl_target->queue.memcpy(d_npart_cell_es_ptr + cell_start,
                                        h_npart_cell_es_ptr + cell_start,
                                        sizeof(INT) * range_cell_count));
      es.push(sycl_target->queue.memcpy(d_npart_cell_ptr + cell_start,
                                        h_npart_cell_ptr + cell_start,
                                        range_cell_count * sizeof(int)));
    }

    this->sub_group_particle_map->create(cell_start, cell_end, h_npart_cell_ptr,
                                         h_npart_cell_es_ptr);

    if ((range_cell_count > 0) && (max_occ > 0)) {
      const std::size_t local_size =
          sycl_target->parameters
              ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
              ->value;
      const std::size_t range_cell = get_next_multiple(max_occ, local_size);

      const auto k_npart_cell = d_npart_cell_ptr;
      const auto k_cell_starts =
          this->sub_group_particle_map->d_cell_starts->ptr;
      const auto k_parent_map = s_parent.d_map_cells_to_particles;
      const auto k_cell_start = this->cell_start;

      es.wait();
      es.push(sycl_target->queue.parallel_for(
          sycl_target->device_limits.validate_nd_range(
              sycl::nd_range<2>(sycl::range<2>(range_cell_count, range_cell),
                                sycl::range<2>(1, local_size))),
          [=](sycl::nd_item<2> idx) {
            const std::size_t index_cell = idx.get_global_id(0) + k_cell_start;
            const std::size_t index_layer = idx.get_global_id(1);
            if (index_layer < k_npart_cell[index_cell]) {
              k_cell_starts[index_cell][index_layer] =
                  k_parent_map.map_loop_layer_to_layer(index_cell, index_layer);
            }
          }));
    }

    es.wait();

    Selection s;
    s.npart_local = es_tmp;
    s.ncell = cell_count;
    s.h_npart_cell = h_npart_cell_ptr;
    s.d_npart_cell = d_npart_cell_ptr;
    s.d_npart_cell_es = d_npart_cell_es_ptr;
    s.d_map_cells_to_particles = {
        this->sub_group_particle_map->d_cell_starts->ptr};
    *created_selection = s;

    pr0.end();
    sycl_target->profile_map.add_region(pr0);
  }
}

} // namespace ParticleSubGroupImplementation

} // namespace NESO::Particles
