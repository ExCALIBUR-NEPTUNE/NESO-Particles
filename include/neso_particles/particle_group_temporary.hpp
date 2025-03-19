#ifndef _NESO_PARTICLES_PARTICLE_GROUP_TEMPORARY_H__
#define _NESO_PARTICLES_PARTICLE_GROUP_TEMPORARY_H__

#include "containers/resource_stack.hpp"
#include "particle_group_impl.hpp"

namespace NESO::Particles {

namespace Private {

struct ParticleGroupTemporaryRSI : ResourceStackInterface<ParticleGroup> {

  DomainSharedPtr domain;
  ParticleSpec *particle_spec;
  SYCLTargetSharedPtr sycl_target;

  virtual inline ParticleGroupSharedPtr construct() override final {
    return std::make_shared<ParticleGroup>(this->domain, *this->particle_spec,
                                           this->sycl_target, true);
  }

  virtual inline void
  free([[maybe_unused]] ParticleGroupSharedPtr &particle_group) override final {
  }

  virtual inline void
  clean(ParticleGroupSharedPtr &particle_group) override final {
    particle_group->clear();
  }
};

} // namespace Private

/**
 * Container to efficiently create and store temporary ParticleGroup instances
 * that have the same properties as a source ParticleGroup.
 */
class ParticleGroupTemporary {
protected:
public:
  /**
   * Get a temporary ParticleGroup containing zero particles with the same
   * SYCLTarget, Domain and particle properties as a source ParticleGroup. This
   * function is collective on the communicator of the ParticleGroup.
   *
   * @param particle_group Source ParticleGroup which specifies the temporary
   * returned ParticleGroup. This source ParticleGroup is unmodified.
   * @returns A temporary ParticleGroup. ParticleGroupTemporary::restore must be
   * called with the original source ParticleGroup and the temporary
   * ParticleGroup that this function returns.
   */
  inline ParticleGroupSharedPtr
  get(const ParticleGroupSharedPtr &particle_group) {
    if (particle_group->resource_stack_particle_group_temporary == nullptr) {

      auto tmp = std::make_shared<Private::ParticleGroupTemporaryRSI>();
      tmp->domain = particle_group->domain;
      tmp->particle_spec = &particle_group->particle_spec;
      tmp->sycl_target = particle_group->sycl_target;
      particle_group->resource_stack_particle_group_temporary =
          std::make_shared<ResourceStack<ParticleGroup>>(tmp);
    }

    auto ptr = std::dynamic_pointer_cast<ResourceStack<ParticleGroup>>(
        particle_group->resource_stack_particle_group_temporary);
    NESOASSERT(ptr != nullptr, "Could not cast ptr.");

    return ptr->get();
  }

  /**
   * Return a previously provided temporary ParticleGroup. This method must be
   * called for all ParticleGroups which were provided by this class. This
   * method is collective on the communicator of the ParticleGroup.
   *
   * @param particle_group Source ParticleGroup originally provided as a
   * specification for the temporary ParticleGroup.
   * @param temporary_particle_group The temporary ParticleGroup which was
   * provided to the calling code by ParticleGroupTemporary::get.
   */
  inline void restore(const ParticleGroupSharedPtr &particle_group,
                      ParticleGroupSharedPtr &temporary_particle_group) {
    auto ptr = std::dynamic_pointer_cast<ResourceStack<ParticleGroup>>(
        particle_group->resource_stack_particle_group_temporary);
    NESOASSERT(ptr != nullptr, "Could not cast ptr.");
    ptr->restore(temporary_particle_group);
  }
};

} // namespace NESO::Particles

#endif
