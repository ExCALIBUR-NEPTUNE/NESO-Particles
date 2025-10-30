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
  INT **map_ptr{nullptr};

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
  int npart_local{0};
  int ncell{0};
  int *h_npart_cell{nullptr};
  int *d_npart_cell{nullptr};
  INT *d_npart_cell_es{nullptr};
  MapLoopLayerToLayer d_map_cells_to_particles;
};

/**
 * Helper function to get the selection map on the host (for testing only).
 *
 * @param selection Selection to get a host representaton of.
 * @returns Indexable [cell][layer] map.
 */
std::vector<std::vector<INT>>
get_host_map_cells_to_particles(SYCLTargetSharedPtr sycl_target,
                                const Selection &selection);

/**
 * Base class for creating sub groups.
 */
class SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  std::shared_ptr<LocalArray<int *>> map_ptrs;
  std::shared_ptr<LocalArray<INT **>> map_cell_to_particles_ptrs;
  std::shared_ptr<SubGroupParticleMap> sub_group_particle_map;
  SubGroupSelectorResourceSharedPtr sub_group_selector_resource;

  // As sub groups can be made from sub groups we need methods to recursively
  // collect the dependencies.
  inline void
  add_parent_dependencies([[maybe_unused]] ParticleGroupSharedPtr parent) {}
  void add_parent_dependencies(std::shared_ptr<ParticleSubGroup> parent);
  void add_parent_dependencies(
      std::shared_ptr<ParticleSubGroupImplementation::SubGroupSelectorBase>
          selector);

  void add_sym_dependency(Sym<INT> sym);
  void add_sym_dependency(Sym<REAL> sym);
  void printing_create_outer_start();
  void printing_create_outer_end();
  void printing_create_inner_start(const bool bool_dats, const bool bool_group);
  void printing_create_inner_end();
  virtual inline void create(Selection *created_selection) = 0;

public:
  // The ParticleGroup this selector operates on.
  ParticleGroupSharedPtr particle_group{nullptr};
  // The ParticleSubGroup this selector operates on.
  std::shared_ptr<ParticleSubGroup> particle_sub_group{nullptr};
  // ParticleDat version tracking.
  ParticleGroup::ParticleDatVersionTracker particle_dat_versions;
  // ParticleGroup version tracking.
  ParticleGroup::ParticleGroupVersion particle_group_version;
  // Is this a selector to a whole particle group.
  bool is_whole_particle_group{false};
  // Has this selector been consumed by a ParticleSubGroup
  bool consumed{false};

  virtual ~SubGroupSelectorBase() {
    if (this->sub_group_selector_resource != nullptr) {

      this->map_ptrs = nullptr;
      this->map_cell_to_particles_ptrs = nullptr;
      this->sub_group_particle_map = nullptr;

#ifdef NESO_PARTICLES_TEST_COMPILATION
      NESOASSERT(this->sub_group_selector_resource->map_ptrs.use_count() == 1,
                 "A reference has been kept.");
      NESOASSERT(this->sub_group_selector_resource->map_cell_to_particles_ptrs
                         .use_count() == 1,
                 "A reference has been kept.");
      NESOASSERT(this->sub_group_selector_resource->sub_group_particle_map
                         .use_count() == 1,
                 "A reference has been kept.");
      NESOASSERT(this->sub_group_selector_resource.use_count() == 1,
                 "A reference has been kept.");
#endif
      this->particle_group->resource_stack_sub_group_resource->restore(
          this->sub_group_selector_resource);
    }
  }
  SubGroupSelectorBase() = default;

  /**
   * @param sym Sym<INT> or Sym<REAL> to test dependency tracking on.
   * @returns True if the dependency tracking contains a Sym.
   */
  template <typename T> inline bool depends_on(Sym<T> sym) {
    return static_cast<bool>(this->particle_dat_versions.count(sym));
  }

  /**
   * @param[in, out] selection Selection to update if required.
   * @returns True if selection was updated.
   */
  bool get(Selection *selection);

  /**
   * @param[in, out] bool_dats Bool to indicate if the invalidation is due to
   * ParticleDat updates.
   * @param[in, out] bool_group Bool to indicate if the invalidation is due to
   * ParticleGroup updates.
   * @returns True if this selector is out of date. A selector is out of date if
   * any of the particle property dependencies have been accessed with a write
   * access descriptor since the last call to get. A selector is out of date if
   * any structural changes have been made, e.g. adding/removing particles,
   * calling cell_move or calling hybrid_move etc.
   */
  bool update_required(bool *bool_dats, bool *bool_group);

  /**
   * Constructor for abstract base type.
   *
   * @param parent ParticleGroup or ParticleSubGroup to use as parent.
   */
  SubGroupSelectorBase(std::shared_ptr<ParticleGroup> parent);

  /**
   * Constructor for abstract base type.
   *
   * @param parent ParticleGroup or ParticleSubGroup to use as parent.
   */
  SubGroupSelectorBase(std::shared_ptr<ParticleSubGroup> parent);
};

typedef std::shared_ptr<SubGroupSelectorBase> SubGroupSelectorBaseSharedPtr;
} // namespace ParticleSubGroupImplementation

} // namespace NESO::Particles

#endif
