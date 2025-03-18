#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_IMPL_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_IMPL_HPP_

#include "particle_sub_group_base.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {
inline ParticleGroupSharedPtr SubGroupSelectorBase::get_particle_group(
    std::shared_ptr<ParticleSubGroup> parent) {
  return parent->get_particle_group();
}

inline void SubGroupSelectorBase::add_parent_dependencies(
    std::shared_ptr<ParticleSubGroup> parent) {
  if (parent != nullptr) {
    for (const auto &dep : parent->selector->particle_dat_versions) {
      this->particle_dat_versions[dep.first] = 0;
    }
  }
}
} // namespace ParticleSubGroupImplementation

} // namespace NESO::Particles

#endif
