#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../particle_group.hpp"

namespace NESO::Particles {
class ParticleSubGroup;

namespace ParticleSubGroupImplementation {

class SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  std::shared_ptr<CellDat<INT>> map_cell_to_particles;

  // Methods to extract the parent ParticleGroup
  inline ParticleGroupSharedPtr
  get_particle_group(std::shared_ptr<ParticleSubGroup> parent);
  inline ParticleGroupSharedPtr
  get_particle_group(std::shared_ptr<ParticleGroup> parent) {
    return parent;
  }

  // As sub groups can be made from sub groups we need methods to recursively
  // collect the dependencies.
  inline void
  add_parent_dependencies([[maybe_unused]] ParticleGroupSharedPtr parent) {}
  inline void add_parent_dependencies(std::shared_ptr<ParticleSubGroup> parent);

  // setup the properties on this base class
  inline void internal_setup_base() {
    auto sycl_target = particle_group->sycl_target;
    const int cell_count = particle_group->domain->mesh->get_cell_count();
    this->map_cell_to_particles =
        std::make_shared<CellDat<INT>>(sycl_target, cell_count, 1);
  }

public:
  // The ParticleGroup this selector operates on.
  ParticleGroupSharedPtr particle_group;
  // ParticleDat version tracking.
  ParticleGroup::ParticleDatVersionTracker particle_dat_versions;
  // ParticleGroup version tracking.
  ParticleGroup::ParticleGroupVersion particle_group_version;

  // The type that describes a selection of particles.
  struct SelectionT {
    int npart_local;
    int ncell;
    int *h_npart_cell;
    int *d_npart_cell;
    INT *d_npart_cell_es;
    INT ***d_map_cells_to_particles;
  };

  virtual ~SubGroupSelectorBase() = default;
  SubGroupSelectorBase() = default;

  virtual inline SelectionT get() = 0;

  template <typename PARENT>
  SubGroupSelectorBase(std::shared_ptr<PARENT> parent)
      : particle_group(get_particle_group(parent)) {}
};

typedef std::shared_ptr<SubGroupSelectorBase> SubGroupSelectorBaseSharedPtr;
} // namespace ParticleSubGroupImplementation

} // namespace NESO::Particles

#endif
