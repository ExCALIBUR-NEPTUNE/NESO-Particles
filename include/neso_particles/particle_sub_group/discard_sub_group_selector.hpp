#ifndef _NESO_PARTICLES_SUB_GROUP_DISCARD_SUB_GROUP_SELECTOR_HPP_
#define _NESO_PARTICLES_SUB_GROUP_DISCARD_SUB_GROUP_SELECTOR_HPP_

#include "particle_sub_group_base.hpp"
#include "particle_sub_group_utility.hpp"
#include "sub_group_selector.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles::ParticleSubGroupImplementation {

/**
 * ParticleSubGroup selector to discard the first N particles in each cell.
 */
class DiscardSubGroupSelector : public SubGroupSelector {
protected:
  inline bool get_parent_is_whole_group(ParticleGroupSharedPtr) { return true; }
  inline bool get_parent_is_whole_group(ParticleSubGroupSharedPtr parent) {
    return parent->is_entire_particle_group();
  }

  bool parent_is_whole_group;
  int num_particles{0};

public:
  /**
   * Create a ParticleSubGroup from a parent by discarding the first n
   * particles in each cell.
   *
   * @param parent Particle(Sub)Group which is the parent.
   * @param num_particles Number of particles to keep from each cell.
   */
  template <typename PARENT>
  DiscardSubGroupSelector(std::shared_ptr<PARENT> parent,
                          const int num_particles)
      : SubGroupSelector(parent), num_particles(num_particles) {

    this->particle_group = get_particle_group(parent);
    this->parent_is_whole_group = this->get_parent_is_whole_group(parent);
    NESOASSERT(num_particles >= 0, "Discarding below zero unsupported.");
  }

  virtual void create(Selection *created_selection) override;
};

extern template DiscardSubGroupSelector::DiscardSubGroupSelector(
    std::shared_ptr<ParticleGroup> parent, const int num_particles);
extern template DiscardSubGroupSelector::DiscardSubGroupSelector(
    std::shared_ptr<ParticleSubGroup> parent, const int num_particles);

} // namespace NESO::Particles::ParticleSubGroupImplementation

namespace NESO::Particles {

class ParticleSubGroup;
class ParticleGroup;

/**
 * Create a ParticleSubGroup from a parent by discarding the first n
 * particles in each cell.
 *
 * @param parent Particle(Sub)Group which is the parent.
 * @param num_particles Number of particles to discard from each cell.
 */
std::shared_ptr<ParticleSubGroup>
particle_sub_group_discard(std::shared_ptr<ParticleGroup> particle_group,
                           const int num_particles);

/**
 * Create a ParticleSubGroup from a parent by discarding the first n
 * particles in each cell.
 *
 * @param parent Particle(Sub)Group which is the parent.
 * @param num_particles Number of particles to discard from each cell.
 */
std::shared_ptr<ParticleSubGroup>
particle_sub_group_discard(std::shared_ptr<ParticleSubGroup> particle_sub_group,
                           const int num_particles);

} // namespace NESO::Particles

#endif
