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

using namespace NESO::Particles;

namespace NESO::Particles {

namespace {

/**
 * TODO
 */
class SubGroupSelector {
protected:
  LocalArray<int> index;
  LocalArray<INT *> ptrs;
  ParticleLoopSharedPtr loop;

public:
  ParticleGroupSharedPtr particle_group;

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
            const INT store_index = k_index(0, 1);
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
   * TODO
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

} // namespace

// This allows the ParticleLoop to access the implementation methods.
template <typename KERNEL, typename... ARGS> class ParticleLoopSubGroup;

/**
 * TODO
 */
class ParticleSubGroup {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS>
  friend class ParticleLoopSubGroup;

protected:
  ParticleGroupSharedPtr particle_group;
  SubGroupSelector selector;

  std::shared_ptr<BufferDeviceHost<INT>> dh_cells;
  std::shared_ptr<BufferDeviceHost<INT>> dh_layers;
  int npart_local;

  template <template <typename> typename T, typename U>
  inline void check_sym_type(T<U> arg) {
    static_assert(std::is_same<T<U>, Sym<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Sym type check failed.");
  }
  template <template <typename> typename T, typename U>
  inline void check_read_access(T<U> arg) {
    static_assert(std::is_same<T<U>, Access::Read<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Read access check failed.");
    check_sym_type(arg.obj);
  }

  ParticleSubGroup(SubGroupSelector selector)
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

public:
  /**
   * TODO
   *
   * arg should be Syms only
   */
  template <typename KERNEL, typename... ARGS>
  ParticleSubGroup(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                   ARGS... args)
      : ParticleSubGroup(SubGroupSelector(particle_group, kernel, args...)) {
    (check_read_access(args), ...);
  }

  inline void create() {
    auto buffers = this->selector.get();
    this->npart_local = std::get<0>(buffers);
    this->dh_cells = std::get<1>(buffers);
    this->dh_layers = std::get<2>(buffers);
  }
};

typedef std::shared_ptr<ParticleSubGroup> ParticleSubGroupSharedPtr;

/**
 * TODO
 */
template <typename KERNEL, typename... ARGS>
class ParticleLoopSubGroup : public ParticleLoop<KERNEL, ARGS...> {

protected:
  ParticleSubGroupSharedPtr particle_sub_group;

  using typename ParticleLoop<KERNEL, ARGS...>::loop_parameter_type;
  using typename ParticleLoop<KERNEL, ARGS...>::kernel_parameter_type;
  using ParticleLoop<KERNEL, ARGS...>::create_loop_args;
  using ParticleLoop<KERNEL, ARGS...>::create_kernel_args;

public:
  ParticleLoopSubGroup(const std::string name,
                       ParticleSubGroupSharedPtr particle_sub_group,
                       KERNEL kernel, ARGS... args)
      : ParticleLoop<KERNEL, ARGS...>(name, particle_sub_group->particle_group,
                                      kernel, args...),
        particle_sub_group(particle_sub_group) {
    this->loop_type = "ParticleLoopSubGroup";
  }
  ParticleLoopSubGroup(ParticleSubGroupSharedPtr particle_sub_group,
                       KERNEL kernel, ARGS... args)
      : ParticleLoop<KERNEL, ARGS...>(particle_sub_group->particle_group,
                                      kernel, args...),
        particle_sub_group(particle_sub_group) {
    this->loop_type = "ParticleLoopSubGroup";
  }

  /**
   *  Launch the ParticleLoop and return. Must be called collectively over the
   *  MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   */
  inline void submit() override {
    NESOASSERT(
        !this->loop_running,
        "ParticleLoop::submit called - but the loop is already submitted.");
    this->loop_running = true;

    auto t0 = profile_timestamp();
    auto position_dat = this->particle_group->position_dat;
    auto d_npart_cell = position_dat->d_npart_cell;
    auto is = this->iteration_set->get();
    auto k_kernel = this->kernel;

    // TODO track remaking better
    this->particle_sub_group->create();
    const INT *d_cells = this->particle_sub_group->dh_cells->d_buffer.ptr;
    const INT *d_layers = this->particle_sub_group->dh_layers->d_buffer.ptr;

    const int nbin = std::get<0>(is);
    this->particle_group->sycl_target->profile_map.inc(
        "ParticleLoop", "Init", 1, profile_elapsed(t0, profile_timestamp()));

    const std::size_t npart_local =
        static_cast<std::size_t>(this->particle_sub_group->npart_local);
    if (npart_local == 0) {
      return;
    }

    sycl::nd_range<1> iteration_set = get_nd_range_1d(npart_local, 256);

    this->event_stack.push(this->particle_group->sycl_target->queue.submit(
        [&](sycl::handler &cgh) {
          loop_parameter_type loop_args;
          create_loop_args(cgh, loop_args);
          cgh.parallel_for<>(iteration_set, [=](sycl::nd_item<1> idx) {
            const std::size_t index = idx.get_global_linear_id();
            if (index < npart_local) {
              const int cellx = static_cast<int>(d_cells[index]);
              const int layerx = static_cast<int>(d_layers[index]);
              kernel_parameter_type kernel_args;
              create_kernel_args(cellx, layerx, loop_args, kernel_args);
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
inline ParticleLoopSharedPtr
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
inline ParticleLoopSharedPtr
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
