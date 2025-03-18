#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_SUB_GROUP_BASE_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_SUB_GROUP_BASE_HPP_

#include "sub_group_selector.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {

// This allows the ParticleLoop to access the implementation methods.
template <typename KERNEL, typename... ARGS> class ParticleLoopSubGroup;

/**
 * A ParticleSubGroup is a container that holds the description of a subset of
 * the particles in a ParticleGroup. For example a sub group could be
 * constructed with all particles of even id. A ParticleSubGroup is a valid
 * iteration set for a ParticleLoop.
 *
 * A ParticleSubGroup with no selector lambda refers to the entire parent
 * ParticleGroup.
 */
class ParticleSubGroup {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS>
  friend class ParticleLoopSubGroup;
  friend class ParticleGroup;
  friend class ParticleSubGroupImplementation::SubGroupSelectorBase;

protected:
  bool is_static;
  ParticleGroupSharedPtr particle_group;
  ParticleSubGroupImplementation::SubGroupSelectorBaseSharedPtr selector;
  ParticleSubGroupImplementation::Selection selection;

  int npart_local;
  bool is_whole_particle_group;

  /**
   * Get the cells and layers of the particles in the sub group (slow)
   */
  inline int get_cells_layers(std::vector<INT> &cells,
                              std::vector<INT> &layers) {
    this->create_if_required();
    cells.resize(this->npart_local);
    layers.resize(this->npart_local);
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();

    auto map_cells_to_particles = get_host_map_cells_to_particles(
        this->particle_group->sycl_target, this->selection);

    INT index = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const int nrow = map_cells_to_particles.at(cellx).size();
      for (int rowx = 0; rowx < nrow; rowx++) {
        cells[index] = cellx;
        const int layerx = map_cells_to_particles.at(cellx).at(rowx);
        layers[index] = layerx;
        index++;
      }
    }
    return npart_local;
  }

  inline void get_cells_layers(INT *d_cells, INT *d_layers);

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
      std::cout << indent << "Recreating ParticleSubGroup: " << (void *)this
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

  inline void create_inner() {
    NESOASSERT(!this->is_whole_particle_group,
               "Explicitly creating the ParticleSubGroup when the sub-group is "
               "the entire ParticleGroup should never be required.");
    this->selection = this->selector->get();
    this->npart_local = this->selection.npart_local;
  }

  inline void create_and_update_cache() {
    create_inner();
    this->particle_group->check_validation(
        this->selector->particle_dat_versions);
    this->particle_group->check_validation(
        this->selector->particle_group_version);
  }

public:
  /**
   * Create a ParticleSubGroup based on a kernel and arguments. The selector
   * kernel must be a lambda which returns true for particles which are in the
   * sub group and false for particles which are not in the sub group. The
   * arguments for the selector kernel must be read access Syms, i.e.
   * Access::read(Sym<T>("name")).
   *
   * For example if A is a ParticleGroup with an INT ParticleProp "ID" that
   * holds particle ids then the following line creates a ParticleSubGroup from
   * the particles with even ids.
   *
   *    auto A_even = std::make_shared<ParticleSubGroup>(
   *      A, [=](auto ID) {
   *        return ((ID[0] % 2) == 0);
   *      },
   *      Access::read(Sym<INT>("ID")));
   *
   * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
   * ParticleSubGroup.
   * @param kernel Lambda function (like a ParticleLoop kernel) that returns
   * true for the particles which should be in the ParticleSubGroup.
   * @param args Arguments in the form of access descriptors wrapping objects
   * to pass to the kernel.
   */
  template <typename PARENT, typename KERNEL, typename... ARGS>
  ParticleSubGroup(std::shared_ptr<PARENT> parent, KERNEL kernel, ARGS... args)
      : ParticleSubGroup(
            std::make_shared<ParticleSubGroupImplementation::SubGroupSelector>(
                parent, kernel, args...)) {}

  /**
   * Create a ParticleSubGroup which is simply a reference/view into an entire
   * ParticleGroup. This constructor creates a sub-group which is equivalent to
   *
   *    auto A_all = std::make_shared<ParticleSubGroup>(
   *      A, [=]() {
   *        return true;
   *      }
   *    );
   *
   * but can make additional optimisations.
   *
   * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
   * ParticleSubGroup.
   */
  ParticleSubGroup(ParticleGroupSharedPtr particle_group)
      : ParticleSubGroup(particle_group, []() { return true; }) {
    this->is_whole_particle_group = true;
  }

  /**
   * Create a ParticleSubGroup directly from a SubGroupSelector. This allows
   * specialsation of the selector via inheritance.
   *
   * @param selector Sub-class of SubGroupSelector.
   */
  ParticleSubGroup(
      ParticleSubGroupImplementation::SubGroupSelectorSharedPtr selector)
      : is_static(false), particle_group(selector->particle_group),
        selector(selector), is_whole_particle_group(false) {}

  /**
   * Create a ParticleSubGroup directly from a SubGroupSelectorBase. This allows
   * specialsation of the selector via inheritance.
   *
   * @param selector Sub-class of SubGroupSelectorBase.
   */
  ParticleSubGroup(
      ParticleSubGroupImplementation::SubGroupSelectorBaseSharedPtr selector)
      : is_static(false), particle_group(selector->particle_group),
        selector(selector), is_whole_particle_group(false) {}

  /**
   * Get and optionally set the static status of the ParticleSubGroup.
   *
   * @param status Optional new static status.
   * @returns Static status.
   */
  inline bool static_status(const std::optional<bool> status = std::nullopt) {
    if (status != std::nullopt) {
      // If the two static values are the same then nothing has to change.
      const bool new_is_static = status.value();
      if (!(this->is_static == new_is_static)) {
        if (new_is_static) {
          // Create the sub group before we disable creating the sub group.
          this->create_if_required();
        }
        this->is_static = new_is_static;
      }
    }
    return this->is_static;
  }

  /**
   * Explicitly re-create the sub group.
   */
  inline void create() { this->create_and_update_cache(); }

  /**
   * Test if a ParticleSubGroup has been invalidated by an operation which
   * irreparably invalidates the internal data structures. This method will
   * always return true for a non-static ParticleSubGroup as non-static
   * ParticleSubGroups are automatically rebuilt as needed.
   *
   * If the ParticleSubGroup is static and this method returns false then the
   * ParticleSubGroup should not be used for any further operations. Note that
   * the implementation calls this method internally to attempt to avoid
   * miss-use of a ParticleSubGroup and hence the user is not expected to
   * routinely call this method.
   *
   * @returns False if a static ParticleSubGroup has been invalidated by an
   * external operation which invalidated the ParticleSubGroup (e.g. moving
   * particles between cells or MPI ranks, adding or removing particles).
   * Otherwise returns True.
   */
  inline bool is_valid() {
    if (this->is_static) {
      return !this->particle_group->check_validation(
          this->selector->particle_group_version, false);
    }
    return true;
  }

  /**
   * Re-create the sub group if required.
   *
   * @returns True if an update occured otherwise false.
   */
  inline bool create_if_required() {
    NESOASSERT(this->is_valid(), "This ParticleSubGroup has been invalidated.");

    if (this->is_whole_particle_group || this->is_static) {
      return false;
    }

    this->printing_create_outer_start();

    const bool bool_dats = this->particle_group->check_validation(
        this->selector->particle_dat_versions);
    const bool bool_group = this->particle_group->check_validation(
        this->selector->particle_group_version);

    if (bool_dats || bool_group) {
      this->printing_create_inner_start(bool_dats, bool_group);
      this->create_inner();
      this->printing_create_inner_end();
      this->printing_create_outer_end();
      return true;
    }

    this->printing_create_outer_end();
    return false;
  }

  /**
   * @return The number of particles currently in the ParticleSubGroup.
   */
  inline INT get_npart_local() {
    if (this->is_whole_particle_group) {
      return this->particle_group->get_npart_local();
    } else {
      this->create_if_required();
      return this->npart_local;
    }
  }

  /**
   * @return The number of particles in a cell of the ParticleSubGroup.
   */
  inline INT get_npart_cell(const int cell) {
    if (this->is_whole_particle_group) {
      return this->particle_group->get_npart_cell(cell);
    } else {
      this->create_if_required();
      return this->selection.h_npart_cell[cell];
    }
  }

  /**
   * @returns The original ParticleGroup this ParticleSubGroup references.
   */
  inline ParticleGroupSharedPtr get_particle_group() {
    return this->particle_group;
  }

  /**
   * @returns True if this ParticleSubGroup references the entirety of the
   * parent ParticleGroup.
   */
  inline bool is_entire_particle_group() {
    return this->is_whole_particle_group;
  }

  /**
   * Create a ParticleSet containing the data from particles held in the
   * ParticleSubGroup. e.g. to Extract the first two particles from the second
   * cell:
   *
   * cells  = [1, 1]
   * layers = [0, 1]
   *
   * @param cells Vector of cell indices of particles to extract.
   * @param cells Vector of layer indices of particles to extract.
   * @returns ParticleSet of particle data.
   */
  inline ParticleSetSharedPtr get_particles(std::vector<INT> &cells,
                                            std::vector<INT> &layers) {
    if (this->is_whole_particle_group) {
      return this->particle_group->get_particles(cells, layers);
    } else {
      this->create_if_required();
      NESOASSERT(cells.size() == layers.size(),
                 "Cells and layers vectors have different sizes.");

      auto sycl_target = this->particle_group->sycl_target;
      auto tmp_buffer =
          get_resource<BufferDeviceHost<INT>,
                       ResourceStackInterfaceBufferDeviceHost<INT>>(
              sycl_target->resource_stack_map,
              ResourceStackKeyBufferDeviceHost<INT>{}, sycl_target);

      const std::size_t num_particles = cells.size();
      tmp_buffer->realloc_no_copy(num_particles * 3);

      INT *d_cells = tmp_buffer->d_buffer.ptr;
      INT *d_layers = d_cells + num_particles;
      INT *d_inner_layers = d_layers + num_particles;

      EventStack es;
      es.push(sycl_target->queue.memcpy(d_cells, cells.data(),
                                        num_particles * sizeof(INT)));
      es.push(sycl_target->queue.memcpy(d_layers, layers.data(),
                                        num_particles * sizeof(INT)));

      const INT num_cells =
          this->particle_group->domain->mesh->get_cell_count();
      for (std::size_t px = 0; px < num_particles; px++) {
        const INT cellx = cells.at(px);
        NESOASSERT((cellx > -1) && (cellx < num_cells),
                   "Cell index not in range.");
        const INT layerx = layers.at(px);
        NESOASSERT((layerx > -1) && (layerx < this->get_npart_cell(cellx)),
                   "Layer index not in range.");
      }

      es.wait();

      auto k_map_cells_to_particles = this->selection.d_map_cells_to_particles;

      auto e0 = sycl_target->queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(num_particles), [=](sycl::id<1> idx) {
          const INT cell = d_cells[idx];
          const INT layer = d_layers[idx];
          const INT inner_layer =
              k_map_cells_to_particles.map_loop_layer_to_layer(cell, layer);
          d_inner_layers[idx] = inner_layer;
        });
      });

      std::vector<INT> inner_layers(num_particles);
      e0.wait_and_throw();
      sycl_target->queue
          .memcpy(inner_layers.data(), d_inner_layers,
                  num_particles * sizeof(INT))
          .wait_and_throw();

      restore_resource(sycl_target->resource_stack_map,
                       ResourceStackKeyBufferDeviceHost<INT>{}, tmp_buffer);

      return this->particle_group->get_particles(cells, inner_layers);
    }
  }

  /**
   * Get a reference to the current selection. This is an advanced method. It is
   * likely that the downstream code is required to call create_if_required
   * directly before calling this method.
   *
   * @returns Reference to the current selection of this instance.
   */
  const ParticleSubGroupImplementation::Selection &get_selection() const {
    return this->selection;
  }

  /**
   *  Print particle data for all particles for the specified ParticleDats.
   *
   *  @param args Sym<REAL> or Sym<INT> instances that indicate which particle
   *  data to print.
   */
  template <typename... T> inline void print(T &&...args) {
    if (this->is_whole_particle_group) {
      return this->particle_group->print(std::forward<T>(args)...);
    } else {
      this->create_if_required();
      SymStore print_spec(std::forward<T>(args)...);

      for (auto &symx : print_spec.syms_real) {
        NESOASSERT(this->particle_group->contains_dat(symx), "Sym not found.");
      }
      for (auto &symx : print_spec.syms_int) {
        NESOASSERT(this->particle_group->contains_dat(symx), "Sym not found.");
      }

      std::cout
          << "==============================================================="
             "================="
          << std::endl;

      const int cell_count =
          this->particle_group->domain->mesh->get_cell_count();

      auto map_cell_to_particles =
          ParticleSubGroupImplementation::get_host_map_cells_to_particles(
              this->particle_group->sycl_target, this->selection);

      for (int cellx = 0; cellx < cell_count; cellx++) {
        const int nrow =
            static_cast<int>(map_cell_to_particles.at(cellx).size());
        if (nrow > 0) {
          std::vector<CellData<REAL>> cell_data_real;
          std::vector<CellData<INT>> cell_data_int;

          for (auto &symx : print_spec.syms_real) {
            auto cell_data = this->particle_group->get_cell(symx, cellx);
            cell_data_real.push_back(cell_data);
          }
          for (auto &symx : print_spec.syms_int) {
            auto cell_data = this->particle_group->get_cell(symx, cellx);
            cell_data_int.push_back(cell_data);
          }

          std::cout << "------- " << cellx << " -------" << std::endl;
          for (auto &symx : print_spec.syms_real) {
            std::cout << "| " << symx.name << " ";
          }
          for (auto &symx : print_spec.syms_int) {
            std::cout << "| " << symx.name << " ";
          }
          std::cout << "|" << std::endl;

          for (int rx = 0; rx < nrow; rx++) {
            const int rowx = map_cell_to_particles.at(cellx).at(rx);
            for (auto &cx : cell_data_real) {
              std::cout << "| ";
              for (int colx = 0; colx < cx->ncol; colx++) {
                std::cout << fixed_width_format((*cx)[colx][rowx]) << " ";
              }
            }
            for (auto &cx : cell_data_int) {
              std::cout << "| ";
              for (int colx = 0; colx < cx->ncol; colx++) {
                std::cout << fixed_width_format((*cx)[colx][rowx]) << " ";
              }
            }

            std::cout << "|" << std::endl;
          }
        }
      }
      std::cout
          << "==============================================================="
             "================="
          << std::endl;
    }
  }
};

typedef std::shared_ptr<ParticleSubGroup> ParticleSubGroupSharedPtr;

} // namespace NESO::Particles

#endif
