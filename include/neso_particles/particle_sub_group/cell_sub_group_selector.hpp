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
    const auto npart_local = this->particle_group->get_npart_local();
    int *d_npart_cell_ptr = this->dh_npart_cell->d_buffer.ptr;
    int *h_npart_cell_ptr = this->dh_npart_cell->h_buffer.ptr;
    const int range_cell_count = this->cell_end - this->cell_start;

    if (this->parent_is_whole_group) {
      std::vector<int> cell_counts(range_cell_count);
      std::vector<INT> cell_counts_es(range_cell_count);
      INT es_tmp = 0;
      for (int cellx = 0; cellx < range_cell_count; cellx++) {
        const int cell = cellx + cell_start;
        const auto total = this->particle_group->get_npart_cell(cell);
        if (this->map_cell_to_particles->nrow.at(cell) < total) {
          this->map_cell_to_particles->set_nrow(cell, total);
        }
        cell_counts[cellx] = total;
        cell_counts_es[cellx] = es_tmp;
        es_tmp += static_cast<INT>(total);
      }

      sycl::event e3, e4;
      e3 = sycl_target->queue.memcpy(this->d_npart_cell_es->ptr + cell_start,
                                     cell_counts_es.data(),
                                     sizeof(INT) * range_cell_count);
      if (cell_end < cell_count) {
        e4 = sycl_target->queue.fill<INT>(this->d_npart_cell_es->ptr + cell_end,
                                          es_tmp, cell_count - cell_end);
      }

      auto e1 = sycl_target->queue.memcpy(h_npart_cell_ptr + cell_start,
                                          cell_counts.data(),
                                          range_cell_count * sizeof(int));
      auto e2 = sycl_target->queue.memcpy(d_npart_cell_ptr + cell_start,
                                          cell_counts.data(),
                                          range_cell_count * sizeof(int));

      this->map_cell_to_particles->wait_set_nrow();

      for (int cellx = 0; cellx < range_cell_count; cellx++) {
        const int cell = cellx + cell_start;
        auto cell_data = this->map_cell_to_particles->get_cell(cell);
        const int cell_npart = cell_counts[cellx];
        for (int rowx = 0; rowx < cell_npart; rowx++) {
          cell_data->at(rowx, 0) = rowx;
        }
        this->map_cell_to_particles->set_cell(cell, cell_data);
      }

      e1.wait_and_throw();
      e2.wait_and_throw();
      e3.wait_and_throw();
      e4.wait_and_throw();

      Selection s;
      s.npart_local = es_tmp;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = this->d_npart_cell_es->ptr;
      s.d_map_cells_to_particles = {this->map_cell_to_particles->device_ptr()};

      return s;

    } else {

      pg_map_layers->realloc_no_copy(npart_local);
      sycl_target->queue
          .fill<int>(d_npart_cell_ptr + cell_start, 0, range_cell_count)
          .wait_and_throw();
      std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
      this->map_ptrs->set(tmp);

      if (range_cell_count == 1) {
        this->loop_0->execute(this->cell_start);
      } else {
        this->loop_0->execute(this->cell_start, this->cell_end);
      }

      this->dh_npart_cell->device_to_host();
      for (int cellx = 0; cellx < range_cell_count; cellx++) {
        const int cell = cellx + this->cell_start;
        const INT nrow_required = h_npart_cell_ptr[cell];
        if (this->map_cell_to_particles->nrow.at(cell) < nrow_required) {
          this->map_cell_to_particles->set_nrow(cell, nrow_required);
        }
      }
      this->map_cell_to_particles->wait_set_nrow();

      if (range_cell_count == 1) {
        this->loop_1->submit(this->cell_start);
      } else {
        this->loop_1->submit(this->cell_start, this->cell_end);
      }

      std::vector<INT> cell_counts_es(range_cell_count);
      INT es_tmp = 0;
      for (int cellx = 0; cellx < range_cell_count; cellx++) {
        const int cell = cellx + cell_start;
        cell_counts_es[cellx] = es_tmp;
        es_tmp += static_cast<INT>(h_npart_cell_ptr[cell]);
      }

      sycl::event e3, e4;
      e3 = sycl_target->queue.memcpy(this->d_npart_cell_es->ptr + cell_start,
                                     cell_counts_es.data(),
                                     sizeof(INT) * range_cell_count);
      if (cell_end < cell_count) {
        e4 = sycl_target->queue.fill<INT>(this->d_npart_cell_es->ptr + cell_end,
                                          es_tmp, cell_count - cell_end);
      }

      this->loop_1->wait();
      e3.wait_and_throw();
      e4.wait_and_throw();

      Selection s;
      s.npart_local = es_tmp;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = this->d_npart_cell_es->ptr;
      s.d_map_cells_to_particles = {this->map_cell_to_particles->device_ptr()};

      return s;
    }
  }
};

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
