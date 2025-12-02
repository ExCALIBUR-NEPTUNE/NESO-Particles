#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_SUB_GROUP_BASE_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_SUB_GROUP_BASE_HPP_

#include "../containers/ephemeral_dats.hpp"
#include "sub_group_selector.hpp"
#include "sub_group_selector_base.hpp"
#include "sub_group_selector_whole_group.hpp"

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
class ParticleSubGroup : public EphemeralDats {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS>
  friend class ParticleLoopSubGroup;
  friend class ParticleGroup;
  friend class ParticleSubGroupImplementation::SubGroupSelectorBase;
  friend inline SymVectorPointerCacheDispatchSharedPtr
  get_sym_vector_cache_dispatch(ParticleGroup *particle_group,
                                ParticleSubGroup *particle_sub_group);

protected:
#ifdef NESO_PARTICLES_TEST_COMPILATION
public:
#endif

  bool is_static;
  ParticleGroupSharedPtr particle_group;
  ParticleSubGroupImplementation::SubGroupSelectorBaseSharedPtr selector{
      nullptr};
  ParticleSubGroupImplementation::Selection selection;

  INT npart_local;
  bool is_whole_particle_group;
  std::int64_t version{1};

  /**
   * Get the cells and layers of the particles in the sub group (slow)
   */
  int get_cells_layers(std::vector<INT> &cells, std::vector<INT> &layers);
  void get_cells_layers(INT *d_cells, INT *d_layers);
  virtual void prepare_ephemeral_dats() override;
  virtual bool invalidate_ephemeral_dats_if_required() override;
  bool create_inner();
  void check_selector(
      ParticleSubGroupImplementation::SubGroupSelectorBaseSharedPtr selector);

public:
  virtual ~ParticleSubGroup() = default;

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
   * @param parent Parent ParticleGroup from which to form ParticleSubGroup.
   */
  ParticleSubGroup(ParticleGroupSharedPtr particle_group);

  /**
   * Create a ParticleSubGroup which is simply a reference/view into an entire
   * ParticleSubGroup. This constructor creates a sub-group which is equivalent
   * to
   *
   *    auto A_all = std::make_shared<ParticleSubGroup>(
   *      A, [=]() {
   *        return true;
   *      }
   *    );
   *
   * but can make additional optimisations.
   *
   * @param parent Parent ParticleSubGroup from which to form ParticleSubGroup.
   */
  ParticleSubGroup(std::shared_ptr<ParticleSubGroup> particle_sub_group);

  /**
   * Create a ParticleSubGroup directly from a SubGroupSelector. This allows
   * specialsation of the selector via inheritance.
   *
   * @param selector Sub-class of SubGroupSelector.
   */
  ParticleSubGroup(
      ParticleSubGroupImplementation::SubGroupSelectorSharedPtr selector);

  /**
   * Create a ParticleSubGroup directly from a SubGroupSelectorBase. This allows
   * specialsation of the selector via inheritance.
   *
   * @param selector Sub-class of SubGroupSelectorBase.
   */
  ParticleSubGroup(
      ParticleSubGroupImplementation::SubGroupSelectorBaseSharedPtr selector);

  /**
   * Get and optionally set the static status of the ParticleSubGroup.
   *
   * @param status Optional new static status.
   * @returns Static status.
   */
  bool static_status(const std::optional<bool> status = std::nullopt);

  /**
   * Explicitly re-create the sub group.
   */
  void create();

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
  bool is_valid();

  /**
   * Re-create the sub group if required.
   *
   * @returns True if an update occured otherwise false.
   */
  bool create_if_required();

  /**
   * @return The number of particles currently in the ParticleSubGroup.
   */
  INT get_npart_local();

  /**
   * @return The number of particles in a cell of the ParticleSubGroup.
   */
  INT get_npart_cell(const int cell);

  /**
   * @returns The original ParticleGroup this ParticleSubGroup references.
   */
  ParticleGroupSharedPtr get_particle_group();

  /**
   * @returns The version of the ParticleSubGroup. The version will always be
   * greater than or equal to one.
   */
  std::int64_t get_version() const;

  /**
   * @returns True if this ParticleSubGroup references the entirety of the
   * parent ParticleGroup.
   */
  bool is_entire_particle_group();

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
  ParticleSetSharedPtr get_particles(std::vector<INT> &cells,
                                     std::vector<INT> &layers);

  /**
   * Get a reference to the current selection. This is an advanced method. It is
   * likely that the downstream code is required to call create_if_required
   * directly before calling this method.
   *
   * @returns Reference to the current selection of this instance.
   */
  const ParticleSubGroupImplementation::Selection &get_selection() const;

protected:
  /**
   *  Print particle data for all particles for the specified ParticleDats.
   *
   *  @param os Output stream to print to.
   *  @param args Sym<REAL> or Sym<INT> instances that indicate which particle
   *  data to print.
   */
  template <typename... T>
  inline void print_inner(std::ostream &os, T &&...args) {
    if (this->is_whole_particle_group) {
      return this->particle_group->print(os, std::forward<T>(args)...);
    } else {
      this->create_if_required();
      SymStore print_spec(std::forward<T>(args)...);

      for (auto &symx : print_spec.syms_real) {
        NESOASSERT(this->particle_group->contains_dat(symx),
                   "Sym not found: " + symx.name);
      }
      for (auto &symx : print_spec.syms_int) {
        NESOASSERT(this->particle_group->contains_dat(symx),
                   "Sym not found. " + symx.name);
      }

      os << "==============================================================="
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

          os << "------- " << cellx << " -------" << std::endl;
          for (auto &symx : print_spec.syms_real) {
            os << "| " << symx.name << " ";
          }
          for (auto &symx : print_spec.syms_int) {
            os << "| " << symx.name << " ";
          }
          os << "|" << std::endl;

          for (int rx = 0; rx < nrow; rx++) {
            const int rowx = map_cell_to_particles.at(cellx).at(rx);
            for (auto &cx : cell_data_real) {
              os << "| ";
              for (int colx = 0; colx < cx->ncol; colx++) {
                os << fixed_width_format((*cx)[colx][rowx]) << " ";
              }
            }
            for (auto &cx : cell_data_int) {
              os << "| ";
              for (int colx = 0; colx < cx->ncol; colx++) {
                os << fixed_width_format((*cx)[colx][rowx]) << " ";
              }
            }

            os << "|" << std::endl;
          }
        }
      }
      os << "==============================================================="
            "================="
         << std::endl;
    }
  }

public:
  /**
   *  Print particle data for all particles for the specified ParticleDats.
   *
   *  @param args Sym<REAL> or Sym<INT> instances that indicate which particle
   *  data to print.
   */
  template <typename... T> inline void print(T &&...args) {
    this->print_inner(std::cout, args...);
  }

  /**
   *  Print particle data for all particles for the specified ParticleDats.
   *
   *  @param os Output stream to print to.
   *  @param args Sym<REAL> or Sym<INT> instances that indicate which particle
   *  data to print.
   */
  template <typename... T> inline void print(std::ostream &os, T &&...args) {
    this->print_inner(os, args...);
  }

  /**
   *  Print particle data for all particles for the specified ParticleDats.
   *
   *  @param os Output stream to print to.
   *  @param args Sym<REAL> or Sym<INT> instances that indicate which particle
   *  data to print.
   */
  template <typename... T> inline void print(std::ofstream &os, T &&...args) {
    this->print_inner(os, args...);
  }
};

typedef std::shared_ptr<ParticleSubGroup> ParticleSubGroupSharedPtr;

} // namespace NESO::Particles

#endif
