#ifndef _NESO_PARTICLES_SUB_GROUP_DISJOINT_UNION_SUB_GROUP_SELECTOR_HPP_
#define _NESO_PARTICLES_SUB_GROUP_DISJOINT_UNION_SUB_GROUP_SELECTOR_HPP_

#include "particle_sub_group_base.hpp"
#include "particle_sub_group_utility.hpp"
#include "sub_group_selector.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {

class ParticleSubGroup;

namespace ParticleSubGroupImplementation {

/**
 * ParticleSubGroup selector which is the union of a set of disjoint selectors.
 */
class DisjointUnionSubGroupSelector : public SubGroupSelector {
protected:
  std::vector<std::shared_ptr<ParticleSubGroup>> parents;

public:
  /**
   * Create a ParticleSubGroup as the union of several particle sub groups.
   *
   * @param parents ParticleSubGroups to unionise.
   */
  DisjointUnionSubGroupSelector(
      std::vector<std::shared_ptr<ParticleSubGroup>> &parents);

  virtual void create(Selection *created_selection) override;
};

} // namespace ParticleSubGroupImplementation

/**
 * @param parents Disjoint ParticleSubGroups to unionise.
 * @param make_static Make the ParticleSubGroup static (default false).
 * @returns Union of input particle sub groups.
 */
std::shared_ptr<ParticleSubGroup> particle_sub_group_disjoint_union(
    std::vector<std::shared_ptr<ParticleSubGroup>> &parents,
    const bool make_static = false);
} // namespace NESO::Particles

#endif
