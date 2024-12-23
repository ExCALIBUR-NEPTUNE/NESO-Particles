#ifndef _PARTICLE_SUB_GROUP_H_
#define _PARTICLE_SUB_GROUP_H_
#include "compute_target.hpp"
#include "loop/particle_loop.hpp"
#include "particle_group.hpp"
#include "typedefs.hpp"
#include <map>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

namespace NESO::Particles {
class ParticleSubGroup;
class ParticleGroup;
namespace ParticleSubGroupImplementation {

/**
 * Class to consume the lambda which selects which particles are to be in a
 * ParticleSubGroup and provide to the ParticleSubGroup a list of cells and
 * layers.
 */
class SubGroupSelector {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  std::shared_ptr<CellDat<INT>> map_cell_to_particles;
  std::shared_ptr<BufferDeviceHost<int>> dh_npart_cell;
  LocalArray<int *> map_ptrs;
  std::shared_ptr<BufferDevice<INT>> d_npart_cell_es;
  ParticleLoopSharedPtr loop_0;
  ParticleLoopSharedPtr loop_1;

  inline ParticleGroupSharedPtr
  get_particle_group(std::shared_ptr<ParticleSubGroup> parent);
  inline ParticleGroupSharedPtr
  get_particle_group(std::shared_ptr<ParticleGroup> parent) {
    return parent;
  }

  template <template <typename> typename T, typename U>
  inline void check_sym_type(T<U> arg) {
    static_assert(std::is_same<T<U>, Sym<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Sym type check failed.");

    // add this sym to the version checker signature
    this->particle_dat_versions[arg] = 0;
  }

  template <typename T> inline void check_sym_type(SymVectorSharedPtr<T> sv) {
    auto dats = sv->get_particle_dats();
    for (auto &dx : dats) {
      this->check_sym_type(dx->sym);
    }
  }

  inline void check_sym_type([[maybe_unused]] ParticleLoopIndex &) {}

  template <template <typename> typename T, typename U>
  inline void check_read_access(T<U> arg) {
    static_assert(std::is_same<T<U>, Access::Read<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Read access check failed.");
    check_sym_type(arg.obj);
  }

  inline void
  add_parent_dependencies([[maybe_unused]] ParticleGroupSharedPtr parent) {}
  inline void add_parent_dependencies(std::shared_ptr<ParticleSubGroup> parent);

  SubGroupSelector() = default;

  template <typename PARENT>
  inline void internal_setup(std::shared_ptr<PARENT> parent) {
    this->add_parent_dependencies(parent);
    this->particle_group_version = 0;

    auto sycl_target = particle_group->sycl_target;
    const int cell_count = particle_group->domain->mesh->get_cell_count();
    this->map_cell_to_particles =
        std::make_shared<CellDat<INT>>(sycl_target, cell_count, 1);
    this->dh_npart_cell =
        std::make_shared<BufferDeviceHost<int>>(sycl_target, cell_count);
    this->map_ptrs = LocalArray<int *>(sycl_target, 2);
    this->d_npart_cell_es =
        std::make_shared<BufferDevice<INT>>(sycl_target, cell_count);

    this->loop_1 = particle_loop(
        "sub_group_selector_1", parent,
        [=](auto loop_index, auto k_map_cell_to_particles, auto k_map_ptrs) {
          const INT particle_linear_index = loop_index.get_local_linear_index();
          const int layer = k_map_ptrs.at(0)[particle_linear_index];
          const bool required = layer > -1;
          if (required) {
            k_map_cell_to_particles.at(layer, 0) = loop_index.layer;
          }
        },
        Access::read(ParticleLoopIndex{}),
        Access::write(this->map_cell_to_particles),
        Access::read(this->map_ptrs));
  }

  inline auto get_particle_group_sub_group_layers() {
    return particle_group->d_sub_group_layers;
  }

public:
  ParticleGroupSharedPtr particle_group;
  ParticleGroup::ParticleDatVersionTracker particle_dat_versions;
  ParticleGroup::ParticleGroupVersion particle_group_version;

  struct SelectionT {
    int npart_local;
    int ncell;
    int *h_npart_cell;
    int *d_npart_cell;
    INT *d_npart_cell_es;
    INT ***d_map_cells_to_particles;
  };

  /**
   * Create a selector based on a kernel and arguments. The selector kernel
   * must be a lambda which returns true for particles which are in the sub
   * group and false for particles which are not in the sub group. The
   * arguments for the selector kernel must be read access Syms, i.e.
   * Access::read(Sym<T>("name")).
   *
   * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
   * ParticleSubGroup.
   * @param kernel Lambda function (like a ParticleLoop kernel) that returns
   * true for the particles which should be in the ParticleSubGroup.
   * @param args Arguments in the form of access descriptors wrapping objects
   * to pass to the kernel.
   */
  template <typename PARENT, typename KERNEL, typename... ARGS>
  SubGroupSelector(std::shared_ptr<PARENT> parent, KERNEL kernel, ARGS... args)
      : particle_group(get_particle_group(parent)) {

    (check_read_access(args), ...);
    this->internal_setup(parent);

    this->loop_0 = particle_loop(
        "sub_group_selector_0", parent,
        [=](auto loop_index, auto k_map_ptrs, auto... user_args) {
          const bool required = kernel(user_args...);
          const INT particle_linear_index = loop_index.get_local_linear_index();
          if (required) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                element_atomic(k_map_ptrs.at(1)[loop_index.cell]);
            const int layer = element_atomic.fetch_add(1);
            k_map_ptrs.at(0)[particle_linear_index] = layer;
          } else {
            k_map_ptrs.at(0)[particle_linear_index] = -1;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs),
        args...);
  }

  virtual ~SubGroupSelector() = default;
  /**
   * Get two BufferDeviceHost objects that hold the cells and layers of the
   * particles which currently are selected by the selector kernel.
   *
   * @returns List of cells and layers of particles in the sub group.
   */
  virtual inline SelectionT get() {
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();
    auto sycl_target = this->particle_group->sycl_target;
    auto pg_map_layers = particle_group->d_sub_group_layers;
    const auto npart_local = this->particle_group->get_npart_local();
    pg_map_layers->realloc_no_copy(npart_local);
    int *d_npart_cell_ptr = this->dh_npart_cell->d_buffer.ptr;
    sycl_target->queue.fill<int>(d_npart_cell_ptr, 0, cell_count)
        .wait_and_throw();
    std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
    this->map_ptrs.set(tmp);

    this->loop_0->execute();

    this->dh_npart_cell->device_to_host();
    int *h_npart_cell_ptr = this->dh_npart_cell->h_buffer.ptr;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const INT nrow_required = h_npart_cell_ptr[cellx];
      if (this->map_cell_to_particles->nrow.at(cellx) < nrow_required) {
        this->map_cell_to_particles->set_nrow(cellx, nrow_required);
      }
    }
    this->map_cell_to_particles->wait_set_nrow();

    this->loop_1->submit();

    std::vector<INT> h_npart_cell_es(cell_count);
    INT total = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      h_npart_cell_es[cellx] = total;
      total += h_npart_cell_ptr[cellx];
    }
    INT *d_npart_cell_es_ptr = this->d_npart_cell_es->ptr;
    sycl_target->queue
        .memcpy(d_npart_cell_es_ptr, h_npart_cell_es.data(),
                cell_count * sizeof(INT))
        .wait();

    this->loop_1->wait();

    SelectionT s;
    s.npart_local = total;
    s.ncell = cell_count;
    s.h_npart_cell = h_npart_cell_ptr;
    s.d_npart_cell = d_npart_cell_ptr;
    s.d_npart_cell_es = d_npart_cell_es_ptr;
    s.d_map_cells_to_particles = this->map_cell_to_particles->device_ptr();
    return s;
  }
};

typedef std::shared_ptr<SubGroupSelector> SubGroupSelectorSharedPtr;

} // namespace ParticleSubGroupImplementation

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
  friend class ParticleSubGroupImplementation::SubGroupSelector;

protected:
  bool is_static;
  ParticleGroupSharedPtr particle_group;
  ParticleSubGroupImplementation::SubGroupSelectorSharedPtr selector;
  ParticleSubGroupImplementation::SubGroupSelector::SelectionT selection;

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

    INT index = 0;
    for (int cellx = 0; cellx < cell_count; cellx++) {
      auto cell_data = this->selector->map_cell_to_particles->get_cell(cellx);
      const int nrow = cell_data->nrow;
      for (int rowx = 0; rowx < nrow; rowx++) {
        cells[index] = cellx;
        const int layerx = cell_data->at(rowx, 0);
        layers[index] = layerx;
        index++;
      }
    }
    return npart_local;
  }

  inline void get_cells_layers(INT *d_cells, INT *d_layers);

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

    const bool bool_dats = this->particle_group->check_validation(
        this->selector->particle_dat_versions);
    const bool bool_group = this->particle_group->check_validation(
        this->selector->particle_group_version);

    if (bool_dats || bool_group) {
      this->create_inner();
      return true;
    }

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
      const int num_particles = cells.size();
      std::vector<INT> inner_layers;
      inner_layers.reserve(num_particles);
      const INT num_cells =
          this->particle_group->domain->mesh->get_cell_count();
      for (int px = 0; px < num_particles; px++) {
        const INT cellx = cells.at(px);
        NESOASSERT((cellx > -1) && (cellx < num_cells),
                   "Cell index not in range.");
        const INT layerx = layers.at(px);
        NESOASSERT((layerx > -1) && (layerx < this->get_npart_cell(cellx)),
                   "Layer index not in range.");

        const INT parent_layer =
            this->selector->map_cell_to_particles->get_value(cellx, layerx, 0);
        inner_layers.push_back(parent_layer);
      }
      return this->particle_group->get_particles(cells, inner_layers);
    }
  }

  /**
   *  Print particle data for all particles for the specified ParticleDats.
   *
   *  @param args Sym<REAL> or Sym<INT> instances that indicate which particle
   *  data to print.
   */
  template <typename... T> inline void print(T... args) {
    if (this->is_whole_particle_group) {
      return this->particle_group->print(args...);
    } else {
      this->create_if_required();
      SymStore print_spec(args...);

      std::cout
          << "==============================================================="
             "================="
          << std::endl;

      const int cell_count =
          this->particle_group->domain->mesh->get_cell_count();
      for (int cellx = 0; cellx < cell_count; cellx++) {
        auto cell_data = this->selector->map_cell_to_particles->get_cell(cellx);
        const int nrow = cell_data->nrow;
        if (nrow > 0) {
          std::vector<CellData<REAL>> cell_data_real;
          std::vector<CellData<INT>> cell_data_int;

          for (auto &symx : print_spec.syms_real) {
            auto cell_data = this->particle_group->particle_dats_real[symx]
                                 ->cell_dat.get_cell(cellx);
            cell_data_real.push_back(cell_data);
          }
          for (auto &symx : print_spec.syms_int) {
            auto cell_data = this->particle_group->particle_dats_int[symx]
                                 ->cell_dat.get_cell(cellx);
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
            const int rowx = cell_data->at(rx, 0);
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

namespace ParticleSubGroupImplementation {

inline ParticleGroupSharedPtr
SubGroupSelector::get_particle_group(std::shared_ptr<ParticleSubGroup> parent) {
  return parent->get_particle_group();
}

inline void SubGroupSelector::add_parent_dependencies(
    std::shared_ptr<ParticleSubGroup> parent) {
  if (parent != nullptr) {
    for (const auto &dep : parent->selector->particle_dat_versions) {
      this->particle_dat_versions[dep.first] = 0;
    }
  }
}

} // namespace ParticleSubGroupImplementation

typedef std::shared_ptr<ParticleSubGroup> ParticleSubGroupSharedPtr;

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
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, KERNEL kernel,
                   ARGS... args) {
  return std::make_shared<ParticleSubGroup>(parent, kernel, args...);
}

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
template <typename PARENT>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent) {
  return std::make_shared<ParticleSubGroup>(parent);
}

/**
 * Create a static ParticleSubGroup based on a kernel and arguments. The
 * selector kernel must be a lambda which returns true for particles which are
 * in the sub group and false for particles which are not in the sub group. The
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
inline ParticleSubGroupSharedPtr
static_particle_sub_group(std::shared_ptr<PARENT> parent, KERNEL kernel,
                          ARGS... args) {
  auto a = std::make_shared<ParticleSubGroup>(parent, kernel, args...);
  a->static_status(true);
  return a;
}

/**
 * Create a static ParticleSubGroup which is simply a reference/view into an
 * entire ParticleGroup. This constructor creates a sub-group which is
 * equivalent to
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
template <typename PARENT>
inline ParticleSubGroupSharedPtr
static_particle_sub_group(std::shared_ptr<PARENT> parent) {
  auto a = std::make_shared<ParticleSubGroup>(parent);
  a->static_status(true);
  return a;
}

/**
 * Derived ParticleLoop type which implements the particle loop over iteration
 * sets defined by ParticleSubGroups.
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoopSubGroup : public ParticleLoop<KERNEL, ARGS...> {

protected:
  ParticleSubGroupSharedPtr particle_sub_group;

  using typename ParticleLoop<KERNEL, ARGS...>::loop_parameter_type;
  using typename ParticleLoop<KERNEL, ARGS...>::kernel_parameter_type;
  using ParticleLoop<KERNEL, ARGS...>::create_loop_args;
  using ParticleLoop<KERNEL, ARGS...>::create_kernel_args;

  virtual inline int get_loop_type_int() override { return 1; }

  inline void setup_subgroup_is(
      ParticleSubGroupImplementation::SubGroupSelector::SelectionT &selection) {
    this->h_npart_cell_lb = selection.h_npart_cell;
    this->d_npart_cell_lb = selection.d_npart_cell;
    this->d_npart_cell_es_lb = selection.d_npart_cell_es;
    this->iteration_set = std::make_unique<
        ParticleLoopImplementation::ParticleLoopBlockIterationSet>(
        this->sycl_target, selection.ncell, this->h_npart_cell_lb,
        this->d_npart_cell_lb);
  }

public:
  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleSubGroup.
   *
   *  @param name Identifier for particle loop.
   *  @param particle_group ParticleSubGroup to execute kernel for all
   * particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopSubGroup(const std::string name,
                       ParticleSubGroupSharedPtr particle_sub_group,
                       KERNEL kernel, ARGS... args)
      : ParticleLoop<KERNEL, ARGS...>(name, particle_sub_group->particle_group,
                                      kernel, args...),
        particle_sub_group(particle_sub_group) {
    this->loop_type = "ParticleLoopSubGroup";
  }

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleSubGroup.
   *
   *  @param particle_group ParticleSubGroup to execute kernel for all
   * particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoopSubGroup(ParticleSubGroupSharedPtr particle_sub_group,
                       KERNEL kernel, ARGS... args)
      : ParticleLoopSubGroup("unnamed_kernel", particle_sub_group, kernel,
                             args...) {}

  /**
   *  Launch the ParticleLoop and return. Must be called collectively over the
   *  MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   *
   *  @param cell Argument for api compatibility.
   */
  inline void submit(const std::optional<int> cell = std::nullopt) override {
    auto t0 = profile_timestamp();
    this->profiling_region_init();

    NESOASSERT(
        (!this->loop_running) || (cell != std::nullopt),
        "ParticleLoop::submit called - but the loop is already submitted.");

    // If the loop is called cell wise asynchronously then the call over cell i
    // could trigger a rebuild on cell i+1
    if (!this->loop_running) {
      this->particle_sub_group->create_if_required();
    }
    this->loop_running = true;

    if (this->iteration_set_is_empty(cell)) {
      return;
    }

    auto &selection = this->particle_sub_group->selection;
    this->setup_subgroup_is(selection);
    auto global_info = this->create_global_info(cell);
    global_info.particle_sub_group = this->particle_sub_group.get();
    this->apply_pre_loop(global_info);

    auto k_kernel = ParticleLoopImplementation::get_kernel(this->kernel);
    auto k_map_cells_to_particles = selection.d_map_cells_to_particles;

    this->profiling_region_metrics(this->iteration_set->iteration_set_size);

    auto &is = this->iteration_set->iteration_set;
    if (cell != std::nullopt) {
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_single_cell(cell.value(),
                                                global_info.local_size, 0);
    } else {
      const std::size_t nbin = this->sycl_target->parameters
                                   ->template get<SizeTParameter>("LOOP_NBIN")
                                   ->value;
      // Num local bytes is already used to compute the local size
      is = this->iteration_set->get_all_cells(nbin, global_info.local_size, 0);
    }
    this->sycl_target->profile_map.inc(
        "ParticleLoopSubGroup", "Init", 1,
        profile_elapsed(t0, profile_timestamp()));

    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            create_loop_args(cgh, loop_args, &global_info);
            cgh.parallel_for<>(
                blockx.loop_iteration_set, [=](sycl::nd_item<2> idx) {
                  std::size_t loop_cell;
                  std::size_t loop_layer;
                  block_device.get_cell_layer(idx, &loop_cell, &loop_layer);
                  const int loop_cellx = static_cast<int>(loop_cell);
                  const int loop_layerx = static_cast<int>(loop_layer);
                  ParticleLoopImplementation::ParticleLoopIteration iterationx;
                  if (block_device.work_item_required(loop_cell, loop_layer)) {
                    const int layer = static_cast<int>(
                        k_map_cells_to_particles[loop_cell][0][loop_layer]);
                    iterationx.local_sycl_index = idx.get_local_id(1);
                    iterationx.cellx = loop_cellx;
                    iterationx.layerx = layer;
                    iterationx.loop_layerx = loop_layerx;
                    kernel_parameter_type kernel_args;
                    create_kernel_args(iterationx, loop_args, kernel_args);
                    Tuple::apply(k_kernel, kernel_args);
                  }
                });
          }));
    }
  }
};

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param particle_group ParticleSubGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(ParticleSubGroupSharedPtr particle_group, KERNEL kernel,
              ARGS... args) {
  if (particle_group->is_entire_particle_group()) {
    return particle_loop(particle_group->get_particle_group(), kernel, args...);
  } else {
    auto p = std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
        particle_group, kernel, args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  }
}

/**
 *  Create a ParticleLoop that executes a kernel for all particles in the
 * ParticleSubGroup.
 *
 *  @param name Identifier for particle loop.
 *  @param particle_group ParticleSubGroup to execute kernel for all particles.
 *  @param kernel Kernel to execute for all particles in the ParticleSubGroup.
 *  @param args The remaining arguments are arguments to be passed to the
 *              kernel. All arguments must be wrapped in an access descriptor
 * type.
 */
template <typename KERNEL, typename... ARGS>
[[nodiscard]] inline ParticleLoopSharedPtr
particle_loop(const std::string name, ParticleSubGroupSharedPtr particle_group,
              KERNEL kernel, ARGS... args) {
  if (particle_group->is_entire_particle_group()) {
    return particle_loop(name, particle_group->get_particle_group(), kernel,
                         args...);
  } else {
    auto p = std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
        name, particle_group, kernel, args...);
    auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
    NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
    return b;
  }
}

inline void ParticleSubGroup::get_cells_layers(INT *d_cells, INT *d_layers) {

  auto lambda_loop = [&](auto iteration_set) {
    particle_loop(
        iteration_set,
        [=](auto index) {
          const INT px = index.get_loop_linear_index();
          d_cells[px] = index.cell;
          d_layers[px] = index.layer;
        },
        Access::read(ParticleLoopIndex{}))
        ->execute();
  };

  if (this->is_entire_particle_group()) {
    lambda_loop(this->particle_group);
  } else {
    lambda_loop(std::shared_ptr<ParticleSubGroup>(
        this, []([[maybe_unused]] auto x) {}));
  }
}

namespace ParticleSubGroupImplementation {

/**
 * ParticleSubGroup selector for a single cell.
 */
class CellSubGroupSelector : public SubGroupSelector {
protected:
  bool parent_is_whole_group;
  int cell;
  inline bool get_parent_is_whole_group(ParticleGroupSharedPtr) { return true; }
  inline bool get_parent_is_whole_group(ParticleSubGroupSharedPtr parent) {
    return parent->is_entire_particle_group();
  }

public:
  template <typename PARENT>
  CellSubGroupSelector(std::shared_ptr<PARENT> parent, const int cell)
      : SubGroupSelector(), cell(cell) {

    this->particle_group = get_particle_group(parent);

    this->check_sym_type(this->particle_group->cell_id_dat->sym);
    this->parent_is_whole_group = this->get_parent_is_whole_group(parent);
    this->internal_setup(parent);
    if (!this->parent_is_whole_group) {
      this->loop_0 = particle_loop(
          "sub_group_selector_0", parent,
          [=](auto loop_index, auto k_map_ptrs) {
            const INT particle_linear_index =
                loop_index.get_local_linear_index();
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                element_atomic(k_map_ptrs.at(1)[loop_index.cell]);
            const int layer = element_atomic.fetch_add(1);
            k_map_ptrs.at(0)[particle_linear_index] = layer;
          },
          Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs));
    }

    const int cell_count = particle_group->domain->mesh->get_cell_count();
    auto e0 = this->particle_group->sycl_target->queue.fill<INT>(
        this->d_npart_cell_es->ptr, 0, cell_count);
    auto e1 = this->particle_group->sycl_target->queue.fill<int>(
        this->dh_npart_cell->d_buffer.ptr, 0, cell_count);
    auto e2 = this->particle_group->sycl_target->queue.fill<int>(
        this->dh_npart_cell->h_buffer.ptr, 0, cell_count);
    e0.wait_and_throw();
    e1.wait_and_throw();
    e2.wait_and_throw();
  }

  virtual inline SelectionT get() override {
    const int cell_count = this->particle_group->domain->mesh->get_cell_count();
    auto sycl_target = this->particle_group->sycl_target;
    auto pg_map_layers = this->get_particle_group_sub_group_layers();
    const auto npart_local = this->particle_group->get_npart_local();
    int *d_npart_cell_ptr = this->dh_npart_cell->d_buffer.ptr;
    int *h_npart_cell_ptr = this->dh_npart_cell->h_buffer.ptr;

    if (this->parent_is_whole_group) {
      const auto total = this->particle_group->get_npart_cell(cell);
      if (this->map_cell_to_particles->nrow.at(this->cell) < total) {
        this->map_cell_to_particles->set_nrow(this->cell, total);
      }
      sycl::event e3;
      if (cell < (cell_count - 1)) {
        e3 = sycl_target->queue.memcpy(this->d_npart_cell_es->ptr + (cell + 1),
                                       &total, sizeof(INT));
      }
      auto e1 =
          sycl_target->queue.fill<int>(h_npart_cell_ptr + cell, (int)total, 1);
      auto e2 =
          sycl_target->queue.fill<int>(d_npart_cell_ptr + cell, (int)total, 1);

      this->map_cell_to_particles->wait_set_nrow();

      auto cell_data = this->map_cell_to_particles->get_cell(cell);
      for (int rowx = 0; rowx < total; rowx++) {
        cell_data->at(rowx, 0) = rowx;
      }
      this->map_cell_to_particles->set_cell(cell, cell_data);

      e1.wait_and_throw();
      e2.wait_and_throw();
      e3.wait_and_throw();

      SelectionT s;
      s.npart_local = total;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = this->d_npart_cell_es->ptr;
      s.d_map_cells_to_particles = this->map_cell_to_particles->device_ptr();

      return s;

    } else {

      pg_map_layers->realloc_no_copy(npart_local);
      sycl_target->queue.fill<int>(d_npart_cell_ptr + cell, 0, 1)
          .wait_and_throw();
      std::vector<int *> tmp = {pg_map_layers->ptr, d_npart_cell_ptr};
      this->map_ptrs.set(tmp);

      this->loop_0->execute(this->cell);

      this->dh_npart_cell->device_to_host();
      const INT nrow_required = h_npart_cell_ptr[this->cell];
      if (this->map_cell_to_particles->nrow.at(this->cell) < nrow_required) {
        this->map_cell_to_particles->set_nrow(this->cell, nrow_required);
      }
      this->map_cell_to_particles->wait_set_nrow();

      this->loop_1->submit(this->cell);

      const INT total = nrow_required;
      if (cell < (cell_count - 1)) {
        sycl_target->queue
            .memcpy(this->d_npart_cell_es->ptr + (cell + 1), &total,
                    sizeof(INT))
            .wait();
      }

      this->loop_1->wait();

      SelectionT s;
      s.npart_local = total;
      s.ncell = cell_count;
      s.h_npart_cell = h_npart_cell_ptr;
      s.d_npart_cell = d_npart_cell_ptr;
      s.d_npart_cell_es = this->d_npart_cell_es->ptr;
      s.d_map_cells_to_particles = this->map_cell_to_particles->device_ptr();

      return s;
    }
  }
};

} // namespace ParticleSubGroupImplementation

/**
 * Create a ParticleSubGroup that selects all particles int a particular cell.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param cell Local cell index to select all particles in.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
template <typename PARENT, typename INT_TYPE,
          std::enable_if_t<std::is_integral<INT_TYPE>::value, bool> = true>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, const INT_TYPE cell,
                   const bool make_static = false) {
  auto selector = std::dynamic_pointer_cast<
      ParticleSubGroupImplementation::SubGroupSelector>(
      std::make_shared<ParticleSubGroupImplementation::CellSubGroupSelector>(
          parent, cell));
  auto group = std::make_shared<ParticleSubGroup>(selector);
  group->static_status(make_static);
  return group;
}

/**
 * Create a ParticleSubGroup that selects all particles int a particular cell.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param cell Local cell index to select all particles in.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
template <typename PARENT>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, const int cell,
                   const bool make_static = false) {
  auto selector = std::dynamic_pointer_cast<
      ParticleSubGroupImplementation::SubGroupSelector>(
      std::make_shared<ParticleSubGroupImplementation::CellSubGroupSelector>(
          parent, cell));
  auto group = std::make_shared<ParticleSubGroup>(selector);
  group->static_status(make_static);
  return group;
}

/**
 * Create a ParticleSubGroup that selects all particles int a particular cell.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param cell Local cell index to select all particles in.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
template <typename PARENT>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, const INT cell,
                   const bool make_static = false) {
  return particle_sub_group(parent, static_cast<int>(cell), make_static);
}

/**
 * Helper function to return the underlying ParticleGroup for a type.
 *
 * @param particle_sub_group ParticleSubGroup.
 * @returns Underlying ParticleGroup.
 */
inline auto get_particle_group(ParticleSubGroupSharedPtr particle_sub_group) {
  return particle_sub_group->get_particle_group();
}

/**
 * Helper function to return the underlying ParticleGroup for a type.
 *
 * @param particle_group ParticleGroup.
 * @returns Underlying ParticleGroup.
 */
inline auto get_particle_group(ParticleGroupSharedPtr particle_group) {
  return particle_group;
}

} // namespace NESO::Particles

#endif
