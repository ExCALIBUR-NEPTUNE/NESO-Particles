#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_WHOLE_GROUP_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_WHOLE_GROUP_HPP_

#include "../particle_group.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {

class ParticleGroup;

namespace ParticleSubGroupImplementation {

/**
 * SubGroupSelector implementation for ParticleSubGroups which are the entire
 * ParticleGroup.
 */
class SubGroupSelectorWholeGroup : public SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

public:
  virtual ~SubGroupSelectorWholeGroup() = default;

  /**
   * @param particle_group ParticleGroup which forms the whole sub group.
   */
  SubGroupSelectorWholeGroup(std::shared_ptr<ParticleGroup> particle_group)
      : SubGroupSelectorBase(particle_group) {
    this->is_whole_particle_group = true;
  }

  /**
   * @returns Selector for whole ParticleGroup.
   */
  virtual inline void create(Selection *created_selection) override {
    auto &cell_id_dat = this->particle_group->cell_id_dat;

    /**
     * If the below selection is made more complicated then the CopySelector
     * also may need revisiting for the case where the selection is the whole
     * ParticleGroup.
     */
    Selection s;
    s.npart_local = this->particle_group->get_npart_local();
    s.ncell = this->particle_group->domain->mesh->get_cell_count();
    s.h_npart_cell = cell_id_dat->h_npart_cell;
    s.d_npart_cell = cell_id_dat->d_npart_cell;
    s.d_npart_cell_es = this->particle_group->dh_npart_cell_es->d_buffer.ptr;
    s.d_map_cells_to_particles = {nullptr};
    *created_selection = s;
  }
};

typedef std::shared_ptr<SubGroupSelectorWholeGroup>
    SubGroupSelectorWholeGroupSharedPtr;

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
