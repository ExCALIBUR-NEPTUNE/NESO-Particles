#ifndef _NESO_PARTICLES_PARTICLE_SUB_GROUP_SUB_GROUP_SELECTOR_RESOURCE_STACK_INTERFACE_HPP_
#define _NESO_PARTICLES_PARTICLE_SUB_GROUP_SUB_GROUP_SELECTOR_RESOURCE_STACK_INTERFACE_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../containers/resource_stack_interface.hpp"
#include "../typedefs.hpp"

namespace NESO::Particles {

/**
 * This is the type that wraps instances that sub-group selectors require.
 */
struct SubGroupSelectorResource {
  std::shared_ptr<CellDat<INT>> map_cell_to_particles;

  ~SubGroupSelectorResource() = default;
  SubGroupSelectorResource() = default;

  SubGroupSelectorResource(std::shared_ptr<CellDat<INT>> map_cell_to_particles)
      : map_cell_to_particles(map_cell_to_particles) {}

  /**
   * Sets the number of rows in the map_cell_to_particles to 0.
   */
  inline void clean() {
    NESOASSERT(this->map_cell_to_particles != nullptr,
               "Clean called on null object.");
    const int ncells = this->map_cell_to_particles->ncells;
    for (int cx = 0; cx < ncells; cx++) {
      this->map_cell_to_particles->set_nrow(cx, 0);
    }
    this->map_cell_to_particles->wait_set_nrow();
  }
};

typedef std::shared_ptr<SubGroupSelectorResource>
    SubGroupSelectorResourceSharedPtr;

/**
 * Defines the interface for ResourceStack to manage instances of
 * SubGroupSelectorResource.
 */
struct SubGroupSelectorResourceStackInterface
    : ResourceStackInterface<SubGroupSelectorResource> {

  SYCLTargetSharedPtr sycl_target;
  int ncells;

  virtual ~SubGroupSelectorResourceStackInterface() = default;

  SubGroupSelectorResourceStackInterface(SYCLTargetSharedPtr sycl_target,
                                         const int ncells)
      : sycl_target(sycl_target), ncells(ncells) {}

  virtual inline SubGroupSelectorResourceSharedPtr construct() override {
    return std::make_shared<SubGroupSelectorResource>(
        std::make_shared<CellDat<INT>>(this->sycl_target, this->ncells, 1));
  }

  virtual inline void
  free([[maybe_unused]] SubGroupSelectorResourceSharedPtr &resource) override {}

  virtual inline void
  clean(SubGroupSelectorResourceSharedPtr &resource) override {
    resource->clean();
  }
};

} // namespace NESO::Particles

#endif
