#ifndef _NESO_PARTICLES_PARTICLE_LOOP_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_H_

#include "../compute_target.hpp"
#include "../containers/tuple.hpp"
#include "../particle_group.hpp"
#include "../particle_spec.hpp"
#include <CL/sycl.hpp>
#include <cstdlib>
#include <typeinfo>

/**
 *  Types and functions relating to access descriptors for loops.
 */
namespace NESO::Particles::Access {

/**
 * Generic base type for an access descriptor around an object of type T.
 */
template <typename T> struct AccessGeneric { T obj; };

/**
 *  Read access descriptor.
 */
template <typename T> struct Read : AccessGeneric<T> {};

/**
 *  Write access descriptor.
 */
template <typename T> struct Write : AccessGeneric<T> {};

/**
 *  Helper function that allows a loop to be constructed with a read-only
 *  parameter passed like:
 *
 *    Access::read(object)
 *
 * @param t Object to pass with read-only access.
 * @returns Access::Read object that wraps passed object.
 */
template <typename T> inline Read<T> read(T t) { return Read<T>{t}; }

/**
 *  Helper function that allows a loop to be constructed with a write
 *  parameter passed like:
 *
 *    Access::write(object)
 *
 * @param t Object to pass with write access.
 * @returns Access::Write object that wraps passed object.
 */
template <typename T> inline Write<T> write(T t) { return Write<T>{t}; }

} // namespace NESO::Particles::Access

/**
 *  Defines the access implementations and types for ParticleDat objects.
 */
namespace NESO::Particles::Access::ParticleDat {

/**
 * Access:ParticleDat::Read<REAL> and Access:ParticleDat::Read<REAL> are the
 * kernel argument types for accessing particle data in a kernel.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for a cell.
  T **ptr;
  /// Stores which particle in the cell this instance refers to.
  int layer;
  T &operator[](const int component) { return ptr[component][this->layer]; }
};

template <typename T> struct Write {
  /// Pointer to underlying data for a cell.
  T **ptr;
  /// Stores which particle in the cell this instance refers to.
  int layer;
  T &operator[](const int component) { return ptr[component][this->layer]; }
};

} // namespace NESO::Particles::Access::ParticleDat

namespace NESO::Particles {

namespace {

/**
 * The LoopParameter types define the type collected from a data structure
 * prior to calling the loop. The loop is responsible for computing the kernel
 * argument from the loop argument. e.g. in the ParticleDat case the
 * LoopParameter is the pointer type that points to the data for all cells,
 * layers and components.
 */
template <typename SPEC> struct LoopParameter { using type = void *; };
/**
 *  Loop parameter for read access of a ParticleDat.
 */
template <typename SPEC> struct LoopParameter<Access::Read<Sym<SPEC>>> {
  using type = SPEC ***;
};
/**
 *  Loop parameter for write access of a ParticleDat.
 */
template <typename SPEC> struct LoopParameter<Access::Write<Sym<SPEC>>> {
  using type = SPEC ***;
};
/**
 *  This is a metafunction which is passed a input data type and returns the
 *  LoopParamter type that corresponds to the input type (by using the structs
 * defined above).
 */
template <class T> using loop_parameter_t = typename LoopParameter<T>::type;

/**
 * The KernelParameter types define the types passed to the kernel for each
 * data structure type for each access descriptor.
 */
template <typename SPEC> struct KernelParameter { using type = void; };
/**
 *  KernelParameter type for read-only access to a ParticleDat.
 */
template <typename SPEC> struct KernelParameter<Access::Read<Sym<SPEC>>> {
  using type = Access::ParticleDat::Read<SPEC>;
};
/**
 *  KernelParameter type for write access to a ParticleDat.
 */
template <typename SPEC> struct KernelParameter<Access::Write<Sym<SPEC>>> {
  using type = Access::ParticleDat::Write<SPEC>;
};
/**
 *  Function to map access descriptor and data structure type to kernel type.
 */
template <class T> using kernel_parameter_t = typename KernelParameter<T>::type;

/**
 * create_kernel_arg is overloaded for each valid pair of access descriptor and
 * data structure which can be passesd to a loop.
 */
/**
 *  Function to create the kernel argument for ParticleDat read access.
 */
template <typename SPEC>
inline void create_kernel_arg(const int cellx, const int layerx, SPEC ***rhs,
                              Access::ParticleDat::Read<SPEC> &lhs) {
  lhs.layer = layerx;
  lhs.ptr = rhs[cellx];
}
/**
 *  Function to create the kernel argument for ParticleDat write access.
 */
template <typename SPEC>
inline void create_kernel_arg(const int cellx, const int layerx, SPEC ***rhs,
                              Access::ParticleDat::Write<SPEC> &lhs) {
  lhs.layer = layerx;
  lhs.ptr = rhs[cellx];
}

} // namespace

/**
 *  ParticleLoop loop type. The particle loop applies the given kernel to all
 *  particles in a ParticleGroup. The kernel must be independent of the
 *  execution order (i.e. parallel and unsequenced in C++ terminology).
 */
template <typename KERNEL, typename... ARGS> class ParticleLoop {

protected:
  /// The types of the parameters for the outside loops.
  using loop_parameter_type = Tuple::Tuple<loop_parameter_t<ARGS>...>;
  /// The types of the arguments passed to the kernel.
  using kernel_parameter_type = Tuple::Tuple<kernel_parameter_t<ARGS>...>;
  /// Tuple of the arguments passed to the ParticleLoop on construction.
  std::tuple<ARGS...> args;

  /// Recursively assemble the tuple args.
  template <size_t INDEX, typename U> inline void unpack_args(U a0) {
    std::get<INDEX>(this->args) = a0;
  }
  template <size_t INDEX, typename U, typename... V>
  inline void unpack_args(U a0, V... args) {
    std::get<INDEX>(this->args) = a0;
    this->unpack_args<INDEX + 1>(args...);
  }

  /// Method to compute access to a particle dat (read)
  template <typename SPEC> inline auto get_loop_arg(Access::Read<Sym<SPEC>> a) {
    auto sym = a.obj;
    return this->particle_group->get_dat(sym)->cell_dat.device_ptr();
  }

  /// Method to compute access to a particle dat (write)
  template <typename SPEC>
  inline auto get_loop_arg(Access::Write<Sym<SPEC>> a) {
    auto sym = a.obj;
    return this->particle_group->get_dat(sym)->cell_dat.device_ptr();
  }
  /// Recursively assemble the outer loop arguments.
  template <size_t INDEX, size_t SIZE, typename PARAM>
  inline void create_loop_args_inner(PARAM &loop_args) {
    if constexpr (INDEX < SIZE) {
      Tuple::get<INDEX>(loop_args) = get_loop_arg(std::get<INDEX>(this->args));
      create_loop_args_inner<INDEX + 1, SIZE>(loop_args);
    }
  }
  inline void create_loop_args(loop_parameter_type &loop_args) {
    create_loop_args_inner<0, sizeof...(ARGS)>(loop_args);
  }

  /// recusively assemble the kernel arguments from the loop arguments
  template <size_t INDEX, size_t SIZE>
  static inline constexpr void
  create_kernel_args_inner(const int cellx, const int layerx,
                           const loop_parameter_type &loop_args,
                           kernel_parameter_type &kernel_args) {

    if constexpr (INDEX < SIZE) {
      auto arg = Tuple::get<INDEX>(loop_args);
      create_kernel_arg(cellx, layerx, arg, Tuple::get<INDEX>(kernel_args));
      create_kernel_args_inner<INDEX + 1, SIZE>(cellx, layerx, loop_args,
                                                kernel_args);
    }
  }

  /// called before kernel execution to assemble the kernel arguments.
  static inline constexpr void
  create_kernel_args(const int cellx, const int layerx,
                     const loop_parameter_type &loop_args,
                     kernel_parameter_type &kernel_args) {

    create_kernel_args_inner<0, sizeof...(ARGS)>(cellx, layerx, loop_args,
                                                 kernel_args);
  }

  ParticleGroupSharedPtr particle_group;
  KERNEL kernel;

public:
  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleGroup.
   *
   *  @param particle_group ParticleGroup to execute kernel for all particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoop(ParticleGroupSharedPtr particle_group, KERNEL kernel,
               ARGS... args)
      : particle_group(particle_group), kernel(kernel) {
    this->unpack_args<0>(args...);
  };

  /**
   *  Launch the ParticleLoop and return. Pushes events onto an event stack
   *  that must be waited on for completion of the ParticleLoop.
   *
   *  @param event_stack EventStack to use for events.
   */
  inline void execute(EventStack &event_stack) {

    auto h_npart_cell = this->particle_group->position_dat->h_npart_cell;
    const int ncell = this->particle_group->position_dat->ncell;

    loop_parameter_type loop_args;
    create_loop_args(loop_args);
    auto k_kernel = this->kernel;

    // For each cell in the mesh we execute a parallel loop over particles.
    for (int cellh = 0; cellh < ncell; cellh++) {
      const int cellx = cellh;
      const int num_particles = h_npart_cell[cellh];

      if (num_particles > 0) {
        // get two iteration sets - a main loop and a peel loop
        auto iteration_set = get_nd_range_peel_1d(num_particles, 256);
        // launch the main loop
        event_stack.push(this->particle_group->sycl_target->queue.submit(
            [&](sycl::handler &cgh) {
              cgh.parallel_for<>(
                  iteration_set.loop_main, [=](sycl::nd_item<1> idx) {
                    const int layerx =
                        static_cast<int>(idx.get_global_linear_id());
                    kernel_parameter_type kernel_args;
                    create_kernel_args(cellx, layerx, loop_args, kernel_args);
                    Tuple::apply(k_kernel, kernel_args);
                  });
            }));
        if (iteration_set.peel_exists) {
          // launch the peel loop
          const std::size_t k_offset = iteration_set.offset;
          event_stack.push(this->particle_group->sycl_target->queue.submit(
              [&](sycl::handler &cgh) {
                cgh.parallel_for<>(
                    iteration_set.loop_peel, [=](sycl::nd_item<1> idx) {
                      const size_t layers =
                          idx.get_global_linear_id() + k_offset;
                      const int layerx = static_cast<int>(layers);
                      // the peel loop requires this masking
                      if (layerx < num_particles) {
                        kernel_parameter_type kernel_args;
                        create_kernel_args(cellx, layerx, loop_args,
                                           kernel_args);
                        Tuple::apply(k_kernel, kernel_args);
                      }
                    });
              }));
        }
      }
    }
  }

  /**
   *  Execute the ParticleLoop and block until execution is complete.
   */
  inline void execute() {
    EventStack es;
    this->execute(es);
    es.wait();
  }
};

} // namespace NESO::Particles

#endif
