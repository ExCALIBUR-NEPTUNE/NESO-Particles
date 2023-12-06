#ifndef _PARTICLE_SUB_GROUP_H_
#define _PARTICLE_SUB_GROUP_H_
#include "compute_target.hpp"
#include "loop/particle_loop.hpp"
#include "particle_group.hpp"
#include "typedefs.hpp"
#include <map>
#include <random>
#include <tuple>
#include <type_traits>

namespace NESO::Particles {

namespace ParticleSubGroupImplementation {

/**
 * Class to consume the lambda which selects which particles are to be in a
 * ParticleSubGroup and provide to the ParticleSubGroup a list of cells and
 * layers.
 */
class SubGroupSelector {
protected:
  LocalArray<int> index;
  LocalArray<INT *> ptrs;
  ParticleLoopSharedPtr loop;

public:
  ParticleGroupSharedPtr particle_group;

  /**
   * Create a selector based on a kernel and arguments. The selector kernel
   * must be a lambda which returns true for particles which are in the sub
   * group and false for particles which are not in the sub group. The
   * arguments for the selector kernel must be read access Syms, i.e.
   * Access::read(Sym<T>("name")).
   *
   * @param particle_group Parent ParticleGroup from which to form
   * ParticleSubGroup.
   * @param kernel Lambda function (like a ParticleLoop kernel) that returns
   * true for the particles which should be in the ParticleSubGroup.
   * @param args Arguments in the form of access descriptors wrapping objects
   * to pass to the kernel.
   */
  template <typename KERNEL, typename... ARGS>
  SubGroupSelector(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                   ARGS... args)
      : particle_group(particle_group) {

    auto sycl_target = particle_group->sycl_target;
    this->index = LocalArray<int>(sycl_target, 1);
    this->ptrs = LocalArray<INT *>(sycl_target, 2);

    this->loop = particle_loop(
        "sub_group_selector", particle_group,
        [=](auto loop_index, auto k_index, auto k_ptrs, auto... user_args) {
          const bool required = kernel(user_args...);
          if (required) {
            // increment the counter by 1 to get the index to store this
            // particle in
            const INT store_index = k_index.fetch_add(0, 1);
            INT *cells = k_ptrs[0];
            INT *layers = k_ptrs[1];
            cells[store_index] = loop_index.cell;
            layers[store_index] = loop_index.layer;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::add(this->index),
        Access::read(this->ptrs), args...);
  }

  /**
   * Get two BufferDeviceHost objects that hold the cells and layers of the
   * particles which currently are selected by the selector kernel.
   *
   * @returns List of cells and layers of particles in the sub group.
   */
  inline std::tuple<int, std::shared_ptr<BufferDeviceHost<INT>>,
                    std::shared_ptr<BufferDeviceHost<INT>>>
  get() {

    const int npart_local = particle_group->get_npart_local();
    auto sycl_target = this->particle_group->sycl_target;

    // TODO - cache these semi globally somewhere?
    auto ptr_cells =
        std::make_shared<BufferDevice<INT>>(sycl_target, npart_local);
    auto ptr_layers =
        std::make_shared<BufferDevice<INT>>(sycl_target, npart_local);

    std::vector<INT *> new_ptrs(2);
    new_ptrs[0] = ptr_cells->ptr;
    new_ptrs[1] = ptr_layers->ptr;
    this->ptrs.set(new_ptrs);

    this->index.fill(0);
    this->loop->execute();

    const int num_particles = this->index.get().at(0);
    auto cells_layers = this->ptrs.get();
    const INT *d_cells = cells_layers.at(0);
    const INT *d_layers = cells_layers.at(1);

    auto dh_cells =
        std::make_shared<BufferDeviceHost<INT>>(sycl_target, num_particles);
    auto dh_layers =
        std::make_shared<BufferDeviceHost<INT>>(sycl_target, num_particles);

    auto k_cells = dh_cells->d_buffer.ptr;
    auto k_layers = dh_layers->d_buffer.ptr;

    const std::size_t size = num_particles * sizeof(INT);
    auto e0 = sycl_target->queue.memcpy(k_cells, d_cells, size);
    auto e1 = sycl_target->queue.memcpy(k_layers, d_layers, size);
    e0.wait_and_throw();
    e1.wait_and_throw();
    dh_cells->device_to_host();
    dh_layers->device_to_host();

    return {num_particles, dh_cells, dh_layers};
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
 */
class ParticleSubGroup {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS>
  friend class ParticleLoopSubGroup;

protected:
  ParticleGroupSharedPtr particle_group;
  ParticleSubGroupImplementation::SubGroupSelector selector;

  std::shared_ptr<BufferDeviceHost<INT>> dh_cells;
  std::shared_ptr<BufferDeviceHost<INT>> dh_layers;
  int npart_local;
  ParticleGroup::ParticleDatVersionTracker particle_dat_versions;

  template <template <typename> typename T, typename U>
  inline void check_sym_type(T<U> arg) {
    static_assert(std::is_same<T<U>, Sym<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Sym type check failed.");

    // add this sym to the version checker signature
    this->particle_dat_versions[arg] = 0;
  }
  template <template <typename> typename T, typename U>
  inline void check_read_access(T<U> arg) {
    static_assert(std::is_same<T<U>, Access::Read<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Read access check failed.");
    check_sym_type(arg.obj);
  }

  ParticleSubGroup(ParticleSubGroupImplementation::SubGroupSelector selector)
      : particle_group(selector.particle_group), selector(selector) {}

  /**
   * Get the cells and layers of the particles in the sub group
   */
  inline int get_cells_layers(std::vector<INT> &cells,
                              std::vector<INT> &layers) {
    this->create();
    cells.resize(npart_local);
    layers.resize(npart_local);
    for (int px = 0; px < npart_local; px++) {
      cells[px] = this->dh_cells->h_buffer.ptr[px];
      layers[px] = this->dh_layers->h_buffer.ptr[px];
    }
    return npart_local;
  }

  inline void create_inner() {
    auto buffers = this->selector.get();
    this->npart_local = std::get<0>(buffers);
    this->dh_cells = std::get<1>(buffers);
    this->dh_layers = std::get<2>(buffers);
  }

  inline void create_and_update_cache() {
    create_inner();
    this->particle_group->check_validation(this->particle_dat_versions);
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
   * @param particle_group Parent ParticleGroup from which to form
   * ParticleSubGroup.
   * @param kernel Lambda function (like a ParticleLoop kernel) that returns
   * true for the particles which should be in the ParticleSubGroup.
   * @param args Arguments in the form of access descriptors wrapping objects
   * to pass to the kernel.
   */
  template <typename KERNEL, typename... ARGS>
  ParticleSubGroup(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                   ARGS... args)
      : ParticleSubGroup(ParticleSubGroupImplementation::SubGroupSelector(
            particle_group, kernel, args...)) {
    (check_read_access(args), ...);
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
    this->create_if_required();
    return this->npart_local;
  }
};

typedef std::shared_ptr<ParticleSubGroup> ParticleSubGroupSharedPtr;

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
    NESOASSERT(cell == std::nullopt,
               "Executing a ParticleLoop over a single cell of a "
               "ParticleSubGroup is not supported.");
    NESOASSERT(
        !this->loop_running,
        "ParticleLoop::submit called - but the loop is already submitted.");
    this->loop_running = true;
    auto global_info = this->create_global_info();

    auto t0 = profile_timestamp();
    auto k_kernel = this->kernel;

    this->particle_sub_group->create_if_required();
    const INT *d_cells = this->particle_sub_group->dh_cells->d_buffer.ptr;
    const INT *d_layers = this->particle_sub_group->dh_layers->d_buffer.ptr;

    this->sycl_target->profile_map.inc(
        "ParticleLoop", "Init", 1, profile_elapsed(t0, profile_timestamp()));

    const std::size_t npart_local =
        static_cast<std::size_t>(this->particle_sub_group->npart_local);
    if (npart_local == 0) {
      return;
    }

    sycl::nd_range<1> iteration_set = get_nd_range_1d(npart_local, 256);

    this->event_stack.push(
        this->sycl_target->queue.submit([&](sycl::handler &cgh) {
          loop_parameter_type loop_args;
          create_loop_args(cgh, loop_args, &global_info);
          cgh.parallel_for<>(iteration_set, [=](sycl::nd_item<1> idx) {
            const std::size_t index = idx.get_global_linear_id();
            if (index < npart_local) {
              const int cellx = static_cast<int>(d_cells[index]);
              const int layerx = static_cast<int>(d_layers[index]);
              kernel_parameter_type kernel_args;
              create_kernel_args(index, cellx, layerx, loop_args, kernel_args);
              Tuple::apply(k_kernel, kernel_args);
            }
          });
        }));
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
  auto p = std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
      particle_group, kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
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
  auto p = std::make_shared<ParticleLoopSubGroup<KERNEL, ARGS...>>(
      name, particle_group, kernel, args...);
  auto b = std::dynamic_pointer_cast<ParticleLoopBase>(p);
  NESOASSERT(b != nullptr, "ParticleLoop pointer cast failed.");
  return b;
}

} // namespace NESO::Particles

#endif
