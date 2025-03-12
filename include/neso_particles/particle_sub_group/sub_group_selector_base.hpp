#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_BASE_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../containers/local_array.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/sub_group_selector_resource_stack_interface.hpp"
#include "sub_group_particle_map.hpp"

namespace NESO::Particles {
class ParticleSubGroup;

namespace ParticleSubGroupImplementation {

/**
 * Device copyable type to map from loop cell and loop layer to the actual layer
 * of the particle.
 */
struct MapLoopLayerToLayer {

  /// This member is public but is not part of any API that should be used
  /// outside of NP - use map_loop_layer_to_layer instead.
  // INT const *RESTRICT const *RESTRICT map_ptr;
  INT **map_ptr;

  /**
   * For a loop cell and loop layer return the layer of the particle.
   *
   * @param loop_cell Cell containing particle in the selection.
   * @param loop_layer Layer of the particle in the selection.
   * @returns Layer of the particle in the cell.
   */
  template <typename T>
  inline INT map_loop_layer_to_layer(const T loop_cell,
                                     const T loop_layer) const {
    return this->map_ptr[loop_cell][loop_layer];
  }
};

/**
 * Host type that describes a selection of particles.
 */
struct Selection {
  int npart_local;
  int ncell;
  int *h_npart_cell;
  int *d_npart_cell;
  INT *d_npart_cell_es;
  MapLoopLayerToLayer d_map_cells_to_particles;
};

/**
 * Helper function to get the selection map on the host (for testing only).
 *
 * @param selection Selection to get a host representaton of.
 * @returns Indexable [cell][layer] map.
 */
inline std::vector<std::vector<INT>>
get_host_map_cells_to_particles(SYCLTargetSharedPtr sycl_target,
                                const Selection &selection) {
  const int cell_count = selection.ncell;
  std::vector<std::vector<INT>> return_map(cell_count);
  std::vector<INT *> d_map_ptrs(cell_count);

  sycl_target->queue
      .memcpy(d_map_ptrs.data(), selection.d_map_cells_to_particles.map_ptr,
              cell_count * sizeof(INT *))
      .wait_and_throw();

  EventStack es;
  for (int cx = 0; cx < cell_count; cx++) {
    return_map.at(cx) = std::vector<INT>(selection.h_npart_cell[cx]);
    es.push(
        sycl_target->queue.memcpy(return_map.at(cx).data(), d_map_ptrs.at(cx),
                                  selection.h_npart_cell[cx] * sizeof(INT)));
  }
  es.wait();
  return return_map;
}

/**
 * Base class for creating sub groups.
 */
class SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  std::shared_ptr<BufferDeviceHost<int>> dh_npart_cell;
  std::shared_ptr<LocalArray<int *>> map_ptrs;
  std::shared_ptr<LocalArray<INT **>> map_cell_to_particles_ptrs;
  std::shared_ptr<BufferHost<INT>> h_npart_cell_es;
  std::shared_ptr<BufferDevice<INT>> d_npart_cell_es;
  std::shared_ptr<SubGroupParticleMap> sub_group_particle_map;

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

    this->sub_group_selector_resource =
        this->particle_group->resource_stack_sub_group_resource->get();
    this->dh_npart_cell = this->sub_group_selector_resource->dh_npart_cell;
    this->map_ptrs = this->sub_group_selector_resource->map_ptrs;
    this->map_cell_to_particles_ptrs =
        this->sub_group_selector_resource->map_cell_to_particles_ptrs;
    this->h_npart_cell_es = this->sub_group_selector_resource->h_npart_cell_es;
    this->d_npart_cell_es = this->sub_group_selector_resource->d_npart_cell_es;
    this->sub_group_particle_map =
        this->sub_group_selector_resource->sub_group_particle_map;
  }

public:
  // The ParticleGroup this selector operates on.
  ParticleGroupSharedPtr particle_group;
  // ParticleDat version tracking.
  ParticleGroup::ParticleDatVersionTracker particle_dat_versions;
  // ParticleGroup version tracking.
  ParticleGroup::ParticleGroupVersion particle_group_version;

  virtual ~SubGroupSelectorBase() {
    if (this->sub_group_selector_resource != nullptr) {
      this->dh_npart_cell = nullptr;
      this->map_ptrs = nullptr;
      this->map_cell_to_particles_ptrs = nullptr;
      this->h_npart_cell_es = nullptr;
      this->d_npart_cell_es = nullptr;
      this->sub_group_particle_map = nullptr;
      this->particle_group->resource_stack_sub_group_resource->restore(
          this->sub_group_selector_resource);
    }
  }
  SubGroupSelectorBase() = default;

  virtual inline Selection get() = 0;

  template <typename PARENT>
  SubGroupSelectorBase(std::shared_ptr<PARENT> parent)
      : particle_group(get_particle_group(parent)) {}
};

typedef std::shared_ptr<SubGroupSelectorBase> SubGroupSelectorBaseSharedPtr;
} // namespace ParticleSubGroupImplementation

} // namespace NESO::Particles

#endif
