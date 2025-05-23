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

  if (this->parent_is_whole_group) {
    INT es_tmp = 0;
    INT max_occ = 0;
    for (int cell = cell_start; cell < cell_end; cell++) {
      const INT total = this->particle_group->get_npart_cell(cell);
      max_occ = std::max(total, max_occ);
      h_npart_cell_ptr[cell] = total;
      h_npart_cell_es_ptr[cell] = es_tmp;
      es_tmp += total;
    }
    for (int cell = cell_end; cell < cell_count; cell++) {
      h_npart_cell_es_ptr[cell] = es_tmp;
    }

    EventStack es;
    es.push(sycl_target->queue.memcpy(d_npart_cell_es_ptr, h_npart_cell_es_ptr,
                                      sizeof(INT) * cell_count));

    if (range_cell_count > 0) {
      es.push(sycl_target->queue.memcpy(d_npart_cell_ptr + cell_start,
                                        h_npart_cell_ptr + cell_start,
                                        range_cell_count * sizeof(int)));
    }

    this->sub_group_particle_map->create(cell_start, cell_end, h_npart_cell_ptr,
                                         h_npart_cell_es_ptr);
    auto h_cell_starts_ptr = this->sub_group_particle_map->h_cell_starts->ptr;
    std::vector<INT> layers(max_occ);
    std::iota(layers.begin(), layers.end(), 0);

    for (int cell = cell_start; cell < cell_end; cell++) {
      const int npart = h_npart_cell_ptr[cell];
      if (npart > 0) {
        es.push(sycl_target->queue.memcpy(h_cell_starts_ptr[cell],
                                          layers.data(), sizeof(INT) * npart));
      }
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

  } else {

    auto pg_map_layers = get_resource<BufferDevice<int>,
                                      ResourceStackInterfaceBufferDevice<int>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
        sycl_target);
    pg_map_layers->realloc_no_copy(npart_local);

    if (range_cell_count > 0) {
      sycl_target->queue
          .fill<int>(d_npart_cell_ptr + cell_start, 0, range_cell_count)
          .wait_and_throw();
    }
    std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
    this->map_ptrs->set(tmp);

    if (range_cell_count == 1) {
      this->loop_0->execute(this->cell_start);
    } else {
      this->loop_0->execute(this->cell_start, this->cell_end);
    }

    if (range_cell_count > 0) {
      sycl_target->queue
          .memcpy(h_npart_cell_ptr + cell_start, d_npart_cell_ptr + cell_start,
                  range_cell_count * sizeof(int))
          .wait_and_throw();
    }

    INT es_tmp = 0;
    for (int cell = cell_start; cell < cell_end; cell++) {
      const INT nrow_required = static_cast<INT>(h_npart_cell_ptr[cell]);
      h_npart_cell_es_ptr[cell] = es_tmp;
      es_tmp += nrow_required;
    }
    for (int cell = cell_end; cell < cell_count; cell++) {
      h_npart_cell_es_ptr[cell] = es_tmp;
    }

    this->sub_group_particle_map->create(cell_start, cell_end, h_npart_cell_ptr,
                                         h_npart_cell_es_ptr);
    auto d_cell_starts_ptr = this->sub_group_particle_map->d_cell_starts->ptr;
    this->map_cell_to_particles_ptrs->set({d_cell_starts_ptr});

    if (range_cell_count == 1) {
      this->loop_1->submit(this->cell_start);
    } else {
      this->loop_1->submit(this->cell_start, this->cell_end);
    }

    EventStack es;
    es.push(sycl_target->queue.memcpy(d_npart_cell_es_ptr, h_npart_cell_es_ptr,
                                      sizeof(INT) * cell_count));

    this->loop_1->wait();

    es.wait();
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, pg_map_layers);

    Selection s;
    s.npart_local = es_tmp;
    s.ncell = cell_count;
    s.h_npart_cell = h_npart_cell_ptr;
    s.d_npart_cell = d_npart_cell_ptr;
    s.d_npart_cell_es = d_npart_cell_es_ptr;
    s.d_map_cells_to_particles = {
        this->sub_group_particle_map->d_cell_starts->ptr};
    *created_selection = s;
  }
}

} // namespace ParticleSubGroupImplementation

} // namespace NESO::Particles
