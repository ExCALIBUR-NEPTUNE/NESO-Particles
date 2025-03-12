#ifndef _NESO_PARTICLES_SUB_GROUP_CELL_SUB_GROUP_SELECTOR_HPP_
#define _NESO_PARTICLES_SUB_GROUP_CELL_SUB_GROUP_SELECTOR_HPP_

#include "particle_sub_group_base.hpp"
#include "sub_group_selector.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

/**
 * ParticleSubGroup selector for a single cell or range of cells.
 */
class CellSubGroupSelector : public SubGroupSelector {
protected:
  bool parent_is_whole_group;
  int cell_start;
  int cell_end;
  inline bool get_parent_is_whole_group(ParticleGroupSharedPtr) { return true; }
  inline bool get_parent_is_whole_group(ParticleSubGroupSharedPtr parent) {
    return parent->is_entire_particle_group();
  }

public:
  /**
   * Create a ParticleSubGroup from a parent by selecting all particles in the
   * parent that are in a range of cell [cell_start, cell_end). Note the last
   * cell is given as a C-style upper bound.
   *
   * @param parent Particle(Sub)Group which is the parent.
   * @param cell_start Starting cell for the cell range.
   * @param cell_end Last cell plus one for the cell range.
   */
  template <typename PARENT>
  CellSubGroupSelector(std::shared_ptr<PARENT> parent, const int cell_start,
                       const int cell_end)
      : SubGroupSelector(), cell_start(cell_start), cell_end(cell_end) {

    this->particle_group = get_particle_group(parent);
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();
    NESOASSERT((cell_start > -1) && (cell_start <= cell_count),
               "Bad cell_start passed.");
    NESOASSERT((cell_end > -1) && (cell_end <= cell_count),
               "Bad cell_end passed.");
    this->check_sym_type(this->particle_group->cell_id_dat->sym);
    this->parent_is_whole_group = this->get_parent_is_whole_group(parent);
    this->internal_setup(parent);
    if (!this->parent_is_whole_group) {
      this->loop_0 = particle_loop(
          "sub_group_selector_0", parent,
          [=](auto loop_index, auto k_map_ptrs) {
            const INT particle_linear_index =
                loop_index.get_local_linear_index();
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                element_atomic(k_map_ptrs.at(1)[loop_index.cell]);
            const int layer = element_atomic.fetch_add(1);
            k_map_ptrs.at(0)[particle_linear_index] = layer;
          },
          Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs));
    }

    auto e0 = this->particle_group->sycl_target->queue.fill<INT>(
        this->d_npart_cell_es->ptr, 0, cell_count);
    auto e1 = this->particle_group->sycl_target->queue.fill<int>(
        this->dh_npart_cell->d_buffer.ptr, 0, cell_count);
    auto e2 = this->particle_group->sycl_target->queue.fill<int>(
        this->dh_npart_cell->h_buffer.ptr, 0, cell_count);
    e0.wait_and_throw();
    e1.wait_and_throw();
    e2.wait_and_throw();
  }

  /**
   * Create a ParticleSubGroup from a parent by selecting all particles in the
   * parent that are in a cell.
   *
   * @param parent Particle(Sub)Group which is the parent.
   * @param cell Cell to select particles from.
   */
  template <typename PARENT>
  CellSubGroupSelector(std::shared_ptr<PARENT> parent, const int cell)
      : CellSubGroupSelector(parent, cell, cell + 1) {
    const int cell_count =
        get_particle_group(parent)->domain->mesh->get_cell_count();
    NESOASSERT((cell > -1) && (cell < cell_count), "Bad cell passed.");
  }

  virtual inline Selection get() override {
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();
    auto sycl_target = this->particle_group->sycl_target;
    auto pg_map_layers = this->get_particle_group_sub_group_layers();

    INT *h_npart_cell_es_ptr = this->h_npart_cell_es->ptr;
    INT *d_npart_cell_es_ptr = this->d_npart_cell_es->ptr;
    int *h_npart_cell_ptr = this->dh_npart_cell->h_buffer.ptr;
    int *d_npart_cell_ptr = this->dh_npart_cell->d_buffer.ptr;

    const auto npart_local = this->particle_group->get_npart_local();
    const int range_cell_count = this->cell_end - this->cell_start;

    if (this->parent_is_whole_group) {
      for (int cell = 0; cell < cell_start; cell++) {
        h_npart_cell_ptr[cell] = 0;
        h_npart_cell_es_ptr[cell] = 0;
      }
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
        h_npart_cell_ptr[cell] = 0;
        h_npart_cell_es_ptr[cell] = es_tmp;
      }

      EventStack es;
      es.push(sycl_target->queue.memcpy(
          d_npart_cell_es_ptr, h_npart_cell_es_ptr, sizeof(INT) * cell_count));

      es.push(sycl_target->queue.memcpy(d_npart_cell_ptr, h_npart_cell_ptr,
                                        cell_count * sizeof(int)));

      this->sub_group_particle_map->create(
          cell_start, cell_end, h_npart_cell_ptr, h_npart_cell_es_ptr);
      auto h_cell_starts_ptr = this->sub_group_particle_map->h_cell_starts->ptr;
      std::vector<INT> layers(max_occ);
      std::iota(layers.begin(), layers.end(), 0);

      for (int cell = cell_start; cell < cell_end; cell++) {
        const int npart = h_npart_cell_ptr[cell];
        es.push(sycl_target->queue.memcpy(h_cell_starts_ptr[cell],
                                          layers.data(), sizeof(INT) * npart));
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

      return s;

    } else {

      pg_map_layers->realloc_no_copy(npart_local);
      sycl_target->queue.fill<int>(d_npart_cell_ptr, 0, cell_count)
          .wait_and_throw();
      std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
      this->map_ptrs->set(tmp);

      if (range_cell_count == 1) {
        this->loop_0->execute(this->cell_start);
      } else {
        this->loop_0->execute(this->cell_start, this->cell_end);
      }

      this->dh_npart_cell->device_to_host();

      for (int cell = 0; cell < cell_start; cell++) {
        h_npart_cell_es_ptr[cell] = 0;
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

      this->sub_group_particle_map->create(
          cell_start, cell_end, h_npart_cell_ptr, h_npart_cell_es_ptr);
      auto d_cell_starts_ptr = this->sub_group_particle_map->d_cell_starts->ptr;
      this->map_cell_to_particles_ptrs->set({d_cell_starts_ptr});

      if (range_cell_count == 1) {
        this->loop_1->submit(this->cell_start);
      } else {
        this->loop_1->submit(this->cell_start, this->cell_end);
      }

      EventStack es;
      es.push(sycl_target->queue.memcpy(
          d_npart_cell_es_ptr, h_npart_cell_es_ptr, sizeof(INT) * cell_count));

      this->loop_1->wait();

      es.wait();
      Selection s;
      s.npart_local = es_tmp;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = d_npart_cell_es_ptr;
      s.d_map_cells_to_particles = {
          this->sub_group_particle_map->d_cell_starts->ptr};

      return s;
    }
  }
};

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
