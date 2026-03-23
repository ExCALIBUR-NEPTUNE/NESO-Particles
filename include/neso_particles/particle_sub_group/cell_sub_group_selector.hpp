#ifndef _NESO_PARTICLES_SUB_GROUP_CELL_SUB_GROUP_SELECTOR_HPP_
#define _NESO_PARTICLES_SUB_GROUP_CELL_SUB_GROUP_SELECTOR_HPP_

#include "particle_loop_sub_group_functions.hpp"
#include "particle_sub_group_base.hpp"
#include "particle_sub_group_utility.hpp"
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
      : SubGroupSelector(parent), cell_start(cell_start), cell_end(cell_end) {

    this->particle_group = get_particle_group(parent);
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();
    NESOASSERT((cell_start > -1) && (cell_start <= cell_count),
               "Bad cell_start passed.");
    NESOASSERT((cell_end > -1) && (cell_end <= cell_count),
               "Bad cell_end passed.");
    NESOASSERT(cell_start <= cell_end, "cell_start > cell_end is invalid");

    this->check_sym_type(this->particle_group->cell_id_dat->sym);
    this->parent_is_whole_group = is_whole_group(parent);

    auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
          d_npart_cell_es_ptr] =
        this->sub_group_particle_map->get_helper_ptrs();

    auto sycl_target = this->particle_group->sycl_target;
    auto e0 = sycl_target->queue.fill<INT>(h_npart_cell_es_ptr, 0, cell_count);
    auto e1 = sycl_target->queue.fill<INT>(d_npart_cell_es_ptr, 0, cell_count);
    auto e2 = sycl_target->queue.fill<int>(h_npart_cell_ptr, 0, cell_count);
    auto e3 = sycl_target->queue.fill<int>(d_npart_cell_ptr, 0, cell_count);

    e0.wait_and_throw();
    e1.wait_and_throw();
    e2.wait_and_throw();
    e3.wait_and_throw();
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

  virtual void create(Selection *created_selection) override;
};

extern template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleGroup> parent, const int cell_start,
    const int cell_end);
extern template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleSubGroup> parent, const int cell_start,
    const int cell_end);

extern template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleGroup> parent, const int cell);
extern template CellSubGroupSelector::CellSubGroupSelector(
    std::shared_ptr<ParticleSubGroup> parent, const int cell);

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
