#ifndef _NESO_PARTICLES_PARTICLE_SUB_GROUP_COPY_SELECTOR_HPP_
#define _NESO_PARTICLES_PARTICLE_SUB_GROUP_COPY_SELECTOR_HPP_

#include "sub_group_selector_base.hpp"

namespace NESO::Particles {

class ParticleSubGroup;

namespace ParticleSubGroupImplementation {

/**
 * Class to consume the lambda which selects which particles are to be in a
 * ParticleSubGroup and provide to the ParticleSubGroup a list of cells and
 * layers.
 */
class CopySelector : public SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  std::shared_ptr<ParticleSubGroup> parent;

public:
  CopySelector(std::shared_ptr<ParticleSubGroup> parent)
      : SubGroupSelectorBase(parent), parent(parent) {
    this->add_parent_dependencies(parent);
  }

  virtual ~CopySelector() = default;

  /**
   * Create the selection.
   *
   * @param[in, out] Selection that describes the ParticleSubGroup.
   */
  virtual void create(Selection *created_selection) override;
};

typedef std::shared_ptr<CopySelector> CopySelectorSharedPtr;

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
