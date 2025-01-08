#ifndef _NESO_PARTICLES_SUB_GROUP_CELL_SUB_GROUP_SELECTOR_HPP_
#define _NESO_PARTICLES_SUB_GROUP_CELL_SUB_GROUP_SELECTOR_HPP_

#include "particle_sub_group_base.hpp"
#include "sub_group_selector.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

/**
 * ParticleSubGroup selector for a single cell.
 */
class CellSubGroupSelector : public SubGroupSelector {
protected:
  bool parent_is_whole_group;
  int cell;
  inline bool get_parent_is_whole_group(ParticleGroupSharedPtr) { return true; }
  inline bool get_parent_is_whole_group(ParticleSubGroupSharedPtr parent) {
    return parent->is_entire_particle_group();
  }

public:
  template <typename PARENT>
  CellSubGroupSelector(std::shared_ptr<PARENT> parent, const int cell)
      : SubGroupSelector(), cell(cell) {

    this->particle_group = get_particle_group(parent);

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

    const int cell_count = particle_group->domain->mesh->get_cell_count();
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

  virtual inline SelectionT get() override {
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();
    auto sycl_target = this->particle_group->sycl_target;
    auto pg_map_layers = this->get_particle_group_sub_group_layers();
    const auto npart_local = this->particle_group->get_npart_local();
    int *d_npart_cell_ptr = this->dh_npart_cell->d_buffer.ptr;
    int *h_npart_cell_ptr = this->dh_npart_cell->h_buffer.ptr;

    if (this->parent_is_whole_group) {
      const auto total = this->particle_group->get_npart_cell(cell);
      if (this->map_cell_to_particles->nrow.at(this->cell) < total) {
        this->map_cell_to_particles->set_nrow(this->cell, total);
      }
      sycl::event e3;
      if (cell < (cell_count - 1)) {
        e3 = sycl_target->queue.memcpy(this->d_npart_cell_es->ptr + (cell + 1),
                                       &total, sizeof(INT));
      }
      auto e1 =
          sycl_target->queue.fill<int>(h_npart_cell_ptr + cell, (int)total, 1);
      auto e2 =
          sycl_target->queue.fill<int>(d_npart_cell_ptr + cell, (int)total, 1);

      this->map_cell_to_particles->wait_set_nrow();

      auto cell_data = this->map_cell_to_particles->get_cell(cell);
      for (int rowx = 0; rowx < total; rowx++) {
        cell_data->at(rowx, 0) = rowx;
      }
      this->map_cell_to_particles->set_cell(cell, cell_data);

      e1.wait_and_throw();
      e2.wait_and_throw();
      e3.wait_and_throw();

      SelectionT s;
      s.npart_local = total;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = this->d_npart_cell_es->ptr;
      s.d_map_cells_to_particles = this->map_cell_to_particles->device_ptr();

      return s;

    } else {

      pg_map_layers->realloc_no_copy(npart_local);
      sycl_target->queue.fill<int>(d_npart_cell_ptr + cell, 0, 1)
          .wait_and_throw();
      std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
      this->map_ptrs.set(tmp);

      this->loop_0->execute(this->cell);

      this->dh_npart_cell->device_to_host();
      const INT nrow_required = h_npart_cell_ptr[this->cell];
      if (this->map_cell_to_particles->nrow.at(this->cell) < nrow_required) {
        this->map_cell_to_particles->set_nrow(this->cell, nrow_required);
      }
      this->map_cell_to_particles->wait_set_nrow();

      this->loop_1->submit(this->cell);

      const INT total = nrow_required;
      if (cell < (cell_count - 1)) {
        sycl_target->queue
            .memcpy(this->d_npart_cell_es->ptr + (cell + 1), &total,
                    sizeof(INT))
            .wait();
      }

      this->loop_1->wait();

      SelectionT s;
      s.npart_local = total;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = this->d_npart_cell_es->ptr;
      s.d_map_cells_to_particles = this->map_cell_to_particles->device_ptr();

      return s;
    }
  }
};

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
