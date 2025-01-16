#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../containers/local_array.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/sub_group_selector_resource_stack_interface.hpp"

namespace NESO::Particles {
class ParticleSubGroup;

namespace ParticleSubGroupImplementation {

class SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  std::shared_ptr<CellDat<INT>> map_cell_to_particles;
  std::shared_ptr<BufferDeviceHost<int>> dh_npart_cell;
  std::shared_ptr<LocalArray<int *>> map_ptrs;
  std::shared_ptr<BufferDevice<INT>> d_npart_cell_es;

  SubGroupSelectorResourceSharedPtr sub_group_selector_resource;

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
    NESOASSERT(this->sub_group_selector_resource == nullptr,
               "Sub-group resource is already allocated somehow.");
    NESOASSERT(this->map_cell_to_particles == nullptr,
               "map_cell_to_particles is not nullptr somehow.");

    this->sub_group_selector_resource =
        this->particle_group->resource_stack_sub_group_resource->get();
    this->map_cell_to_particles =
        this->sub_group_selector_resource->map_cell_to_particles;
    this->dh_npart_cell = this->sub_group_selector_resource->dh_npart_cell;
    this->map_ptrs = this->sub_group_selector_resource->map_ptrs;
    this->d_npart_cell_es = this->sub_group_selector_resource->d_npart_cell_es;
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

  virtual ~SubGroupSelectorBase() {
    if (this->sub_group_selector_resource != nullptr) {
      this->map_cell_to_particles = nullptr;
      this->dh_npart_cell = nullptr;
      this->map_ptrs = nullptr;
      this->d_npart_cell_es = nullptr;
      this->particle_group->resource_stack_sub_group_resource->restore(
          this->sub_group_selector_resource);
    }
  }
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
