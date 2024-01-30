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

public:
  ParticleGroupSharedPtr particle_group;

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

    auto sycl_target = particle_group->sycl_target;
    const int cell_count = particle_group->domain->mesh->get_cell_count();
    this->map_cell_to_particles =
        std::make_shared<CellDat<INT>>(sycl_target, cell_count, 1);
    this->dh_npart_cell =
        std::make_shared<BufferDeviceHost<int>>(sycl_target, cell_count);
    this->map_ptrs = LocalArray<int *>(sycl_target, 2);
    this->d_npart_cell_es =
        std::make_shared<BufferDevice<INT>>(sycl_target, cell_count);

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

  /**
   * Get two BufferDeviceHost objects that hold the cells and layers of the
   * particles which currently are selected by the selector kernel.
   *
   * @returns List of cells and layers of particles in the sub group.
   */
  inline SelectionT get() {

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

protected:
  bool is_static;
  ParticleGroupSharedPtr particle_group;
  ParticleSubGroupImplementation::SubGroupSelector selector;
  ParticleSubGroupImplementation::SubGroupSelector::SelectionT selection;

  int npart_local;
  ParticleGroup::ParticleDatVersionTracker particle_dat_versions;
  bool is_whole_particle_group;

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

  ParticleSubGroup(ParticleSubGroupImplementation::SubGroupSelector selector)
      : particle_group(selector.particle_group), selector(selector),
        is_whole_particle_group(false), is_static(false) {}

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
      auto cell_data = this->selector.map_cell_to_particles->get_cell(cellx);
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

    this->selection = this->selector.get();
    this->npart_local = this->selection.npart_local;
  }

  inline void create_and_update_cache() {
    create_inner();
    this->particle_group->check_validation(this->particle_dat_versions);
  }

  inline void add_parent_dependencies(ParticleGroupSharedPtr parent) {}
  inline void
  add_parent_dependencies(std::shared_ptr<ParticleSubGroup> parent) {
    for (const auto dep : parent->particle_dat_versions) {
      this->particle_dat_versions[dep.first] = 0;
    }
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
      : ParticleSubGroup(ParticleSubGroupImplementation::SubGroupSelector(
            parent, kernel, args...)) {
    (check_read_access(args), ...);
    this->add_parent_dependencies(parent);
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
  ParticleSubGroup(ParticleGroupSharedPtr particle_group)
      : ParticleSubGroup(particle_group, []() { return true; }) {
    this->is_whole_particle_group = true;
  }

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
   * Re-create the sub group if required.
   *
   * @returns True if an update occured otherwise false.
   */
  inline bool create_if_required() {
    if (this->is_whole_particle_group || this->is_static) {
      return false;
    }

    if (this->particle_group->check_validation(this->particle_dat_versions)) {
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
};

namespace ParticleSubGroupImplementation {

inline ParticleGroupSharedPtr
SubGroupSelector::get_particle_group(std::shared_ptr<ParticleSubGroup> parent) {
  return parent->get_particle_group();
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
    this->iteration_set =
        std::make_unique<ParticleLoopImplementation::ParticleLoopIterationSet>(
            1, selection.ncell, this->h_npart_cell_lb);
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

    auto global_info = this->create_global_info();
    global_info.starting_cell = (cell == std::nullopt) ? 0 : cell.value();

    auto k_kernel = this->kernel;
    auto k_npart_cell_lb = this->d_npart_cell_lb;
    auto k_map_cells_to_particles = selection.d_map_cells_to_particles;

    auto is = this->iteration_set->get(cell);
    const int nbin = std::get<0>(is);

    this->sycl_target->profile_map.inc(
        "ParticleLoopSubGroup", "Init", 1,
        profile_elapsed(t0, profile_timestamp()));

    for (int binx = 0; binx < nbin; binx++) {
      sycl::nd_range<2> ndr = std::get<1>(is).at(binx);
      const size_t cell_offset = std::get<2>(is).at(binx);
      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            create_loop_args(cgh, loop_args, &global_info);
            cgh.parallel_for<>(ndr, [=](sycl::nd_item<2> idx) {
              const std::size_t index = idx.get_global_linear_id();
              const size_t cellxs = idx.get_global_id(0) + cell_offset;
              const size_t layerxs = idx.get_global_id(1);
              const int cellx = static_cast<int>(cellxs);
              const int loop_layerx = static_cast<int>(layerxs);
              ParticleLoopImplementation::ParticleLoopIteration iterationx;
              if (loop_layerx < k_npart_cell_lb[cellx]) {
                const int layerx = static_cast<int>(
                    k_map_cells_to_particles[cellxs][0][layerxs]);
                kernel_parameter_type kernel_args;
                iterationx.index = index;
                iterationx.cellx = cellx;
                iterationx.layerx = layerx;
                iterationx.loop_layerx = loop_layerx;
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
    lambda_loop(std::shared_ptr<ParticleSubGroup>(this, [](auto x) {}));
  }
}

} // namespace NESO::Particles

#endif
