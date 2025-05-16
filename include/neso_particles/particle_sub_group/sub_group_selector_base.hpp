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
    if (selection.h_npart_cell[cx] > 0) {
      es.push(
          sycl_target->queue.memcpy(return_map.at(cx).data(), d_map_ptrs.at(cx),
                                    selection.h_npart_cell[cx] * sizeof(INT)));
    }
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
  std::shared_ptr<LocalArray<int *>> map_ptrs;
  std::shared_ptr<LocalArray<INT **>> map_cell_to_particles_ptrs;
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

  inline void add_sym_dependency(Sym<INT> sym) {
    this->particle_dat_versions[sym] = 0;
  }
  inline void add_sym_dependency(Sym<REAL> sym) {
    this->particle_dat_versions[sym] = 0;
  }

  inline void printing_create_outer_start() {
    if (this->particle_group->debug_sub_group_create) {
      if (!this->particle_group->debug_sub_group_indent) {
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Testing recreation criterion: " << (void *)this
                  << std::endl;
      }
      this->particle_group->debug_sub_group_indent += 4;
    }
  }

  inline void printing_create_outer_end() {
    if (this->particle_group->debug_sub_group_create) {
      this->particle_group->debug_sub_group_indent -= 4;
    }
  }

  inline void printing_create_inner_start(const bool bool_dats,
                                          const bool bool_group) {
    if (this->particle_group->debug_sub_group_create) {

      std::string indent(this->particle_group->debug_sub_group_indent, ' ');
      std::cout << indent << "Recreating Selector: " << (void *)this
                << " reason_dats: " << bool_dats
                << " reason_group: " << bool_group << std::endl;
    }
  }

  inline void printing_create_inner_end() {
    if (this->particle_group->debug_sub_group_create) {
      if (!this->particle_group->debug_sub_group_indent) {
        std::cout << std::string(80, '-') << std::endl;
      }
    }
  }

  virtual inline void create(Selection *created_selection) = 0;

public:
  // The ParticleGroup this selector operates on.
  ParticleGroupSharedPtr particle_group;
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
  inline bool get(Selection *selection) {

    this->printing_create_outer_start();

    const bool bool_dats =
        this->particle_group->check_validation(this->particle_dat_versions);
    const bool bool_group =
        this->particle_group->check_validation(this->particle_group_version);

    if (bool_dats || bool_group) {
      this->printing_create_inner_start(bool_dats, bool_group);
      this->create(selection);
      this->printing_create_inner_end();
      this->printing_create_outer_end();
      return true;
    }

    this->printing_create_outer_end();
    return false;
  }

  /**
   * @returns True if this selector is out of date. A selector is out of date if
   * any of the particle property dependencies have been accessed with a write
   * access descriptor since the last call to get. A selector is out of date if
   * any structural changes have been made, e.g. adding/removing particles,
   * calling cell_move or calling hybrid_move etc.
   */
  inline bool update_required() {

    const bool bool_dats = this->particle_group->check_validation(
        this->particle_dat_versions, false);
    const bool bool_group = this->particle_group->check_validation(
        this->particle_group_version, false);

    return bool_dats || bool_group;
  }

  /**
   * Constructor for abstract base type.
   *
   * @param parent ParticleGroup or ParticleSubGroup to use as parent.
   */
  template <typename PARENT>
  SubGroupSelectorBase(std::shared_ptr<PARENT> parent)
      : particle_group(get_particle_group(parent)), particle_group_version(0) {
    this->add_parent_dependencies(parent);

    NESOASSERT(this->sub_group_selector_resource == nullptr,
               "Sub-group resource is already allocated somehow.");
    this->sub_group_selector_resource =
        this->particle_group->resource_stack_sub_group_resource->get();
    this->map_ptrs = this->sub_group_selector_resource->map_ptrs;
    this->map_cell_to_particles_ptrs =
        this->sub_group_selector_resource->map_cell_to_particles_ptrs;
    this->sub_group_particle_map =
        this->sub_group_selector_resource->sub_group_particle_map;
  }
};

typedef std::shared_ptr<SubGroupSelectorBase> SubGroupSelectorBaseSharedPtr;
} // namespace ParticleSubGroupImplementation

} // namespace NESO::Particles

#endif
