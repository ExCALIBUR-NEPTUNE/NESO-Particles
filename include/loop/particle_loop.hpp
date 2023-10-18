#ifndef _NESO_PARTICLES_PARTICLE_LOOP_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_H_

#include "../compute_target.hpp"
#include "../containers/local_array.hpp"
#include "../containers/tuple.hpp"
#include "../particle_dat.hpp"
#include "../particle_group.hpp"
#include "../particle_spec.hpp"
#include <CL/sycl.hpp>
#include <cstdlib>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

using namespace NESO::Particles;

namespace {

/**
 * For a set of cells containing particles create several sycl::nd_range
 * instances which cover the iteration space of all particles. This exists to
 * create an iteration set over all particles which is blocked, to reduce the
 * number of kernel launches, and reasonably robust to non-uniform.
 */
struct ParticleLoopIterationSet {

  /// The number of blocks of cells.
  const int nbin;
  /// The number of cells.
  const int ncell;
  /// Host accessible pointer to the number of particles in each cell.
  int *h_npart_cell;
  /// Container to store the sycl::nd_ranges.
  std::vector<sycl::nd_range<2>> iteration_set;
  /// Offsets to add to the cell index to map to the correct cell.
  std::vector<std::size_t> cell_offsets;

  /**
   *  Creates iteration set creator for a given set of cell particle counts.
   *
   *  @param nbin Number of blocks of cells.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   */
  ParticleLoopIterationSet(const int nbin, const int ncell, int *h_npart_cell)
      : nbin(std::min(ncell, nbin)), ncell(ncell), h_npart_cell(h_npart_cell) {
    this->iteration_set.reserve(nbin);
    this->cell_offsets.resize(nbin);
  }

  /**
   *  Create and return an iteration set which is formed as nbin
   *  sycl::nd_ranges.
   *
   *  @param local_size Optional size of SYCL work groups.
   *  @returns Tuple containing: Number of bins, sycl::nd_ranges, cell index
   *  offsets.
   */
  inline std::tuple<int, std::vector<sycl::nd_range<2>> &,
                    std::vector<std::size_t> &>
  get(const size_t local_size = 256) {
    this->iteration_set.clear();
    this->iteration_set.reserve(nbin);

    for (int binx = 0; binx < nbin; binx++) {
      int start, end;
      get_decomp_1d(nbin, ncell, binx, &start, &end);
      const int bin_width = end - start;
      this->cell_offsets[binx] = static_cast<std::size_t>(start);
      int cell_maxi = 0;
      for (int cellx = start; cellx < end; cellx++) {
        cell_maxi = std::max(cell_maxi, h_npart_cell[cellx]);
      }
      const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                    static_cast<long long>(local_size));
      const std::size_t outer_size =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1)) *
          local_size;

      this->iteration_set.emplace_back(
          sycl::nd_range<2>(sycl::range<2>(bin_width, outer_size),
                            sycl::range<2>(1, local_size)));
    }
    return {this->nbin, this->iteration_set, this->cell_offsets};
  }
};

} // namespace

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
 *  Atomic add access descriptor.
 */
template <typename T> struct Add : AccessGeneric<T> {};

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

/**
 *  Helper function that allows a loop to be constructed with a write
 *  parameter which is atomic addition passed like:
 *
 *    Access::add(object)
 *
 * @param t Object to pass with atomic add access.
 * @returns Access::Add object that wraps passed object.
 */
template <typename T> inline Add<T> add(T t) { return Add<T>{t}; }

} // namespace NESO::Particles::Access

/**
 *  Defines the access implementations and types for ParticleDat objects.
 */
namespace NESO::Particles::Access::ParticleDat {

/**
 * Access:ParticleDat::Read<T> and Access:ParticleDat::Read<T> are the
 * kernel argument types for accessing particle data in a kernel.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for a cell.
  T const *const *ptr;
  /// Stores which particle in the cell this instance refers to.
  int layer;
  T const &operator[](const int component) {
    return ptr[component][this->layer];
  }
};

template <typename T> struct Write {
  /// Pointer to underlying data for a cell.
  T **ptr;
  /// Stores which particle in the cell this instance refers to.
  int layer;
  T &operator[](const int component) { return ptr[component][this->layer]; }
};

} // namespace NESO::Particles::Access::ParticleDat

/**
 *  Defines the access implementations and types for LocalArray objects.
 */
namespace NESO::Particles::Access::LocalArray {

/**
 * Access:LocalArray::Read<T> and Access:LocalArray::Add<T> are the
 * kernel argument types for accessing LocalArray data in a kernel.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for the array.
  Read() = default;
  T const *ptr;
  const T &operator[](const int component) { return ptr[component]; }
};

/**
 * Access:LocalArray::Read<T> and Access:LocalArray::Add<T> are the
 * kernel argument types for accessing LocalArray data in a kernel.
 */
template <typename T> struct Add {
  /// Pointer to underlying data for the array.
  Add() = default;
  T *ptr;
  T operator()(const int component, const T value) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[component]);
    return element_atomic.fetch_add(value);
  }
};

} // namespace NESO::Particles::Access::LocalArray

namespace NESO::Particles {

namespace {

/**
 * The LoopParameter types define the type collected from a data structure
 * prior to calling the loop. The loop is responsible for computing the kernel
 * argument from the loop argument. e.g. in the ParticleDat case the
 * LoopParameter is the pointer type that points to the data for all cells,
 * layers and components.
 */
template <typename T> struct LoopParameter { using type = void *; };
/**
 *  Loop parameter for read access of a ParticleDat.
 */
template <typename T> struct LoopParameter<Access::Read<Sym<T>>> {
  using type = T ***;
};
/**
 *  Loop parameter for write access of a ParticleDat.
 */
template <typename T> struct LoopParameter<Access::Write<Sym<T>>> {
  using type = T ***;
};
/**
 *  Loop parameter for read access of a LocalArray.
 */
template <typename T> struct LoopParameter<Access::Read<LocalArray<T>>> {
  using type = T const *;
};
/**
 *  Loop parameter for add access of a LocalArray.
 */
template <typename T> struct LoopParameter<Access::Add<LocalArray<T>>> {
  using type = T *;
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
template <typename T> struct KernelParameter { using type = void; };
/**
 *  KernelParameter type for read-only access to a ParticleDat.
 */
template <typename T> struct KernelParameter<Access::Read<Sym<T>>> {
  using type = Access::ParticleDat::Read<T>;
};
/**
 *  KernelParameter type for write access to a ParticleDat.
 */
template <typename T> struct KernelParameter<Access::Write<Sym<T>>> {
  using type = Access::ParticleDat::Write<T>;
};
/**
 *  KernelParameter type for read access to a LocalArray.
 */
template <typename T> struct KernelParameter<Access::Read<LocalArray<T>>> {
  using type = Access::LocalArray::Read<T>;
};
/**
 *  KernelParameter type for add access to a LocalArray.
 */
template <typename T> struct KernelParameter<Access::Add<LocalArray<T>>> {
  using type = Access::LocalArray::Add<T>;
};
/**
 *  Function to map access descriptor and data structure type to kernel type.
 */
template <class T> using kernel_parameter_t = typename KernelParameter<T>::type;

} // namespace

// clang-format off
/**
 *  ParticleLoop loop type. The particle loop applies the given kernel to all
 *  particles in a ParticleGroup. The kernel must be independent of the
 *  execution order (i.e. parallel and unsequenced in C++ terminology).
 *
 *  Data Structure  | Access Descriptors | Kernel Argument Type | Notes |
 *  --------------  | ------------------ | -------------------- | ----- |
 *  ParticleDat<T> | Read, Write | Access::ParticleDat::Read<T>, Access::ParticleDat::Write<T> | Loop is called with the Sym<T>, e.g Access::read(Sym<T>("A")) |
 *  LocalArray<T>  | Read, Add | Access::LocalArray::Read<T>, Access::LocalArray::Add<T> | Loop is called with the array, e.g LocalArray l0(...), Access::read(l0) |
 *
 */
// clang-format on
template <typename KERNEL, typename... ARGS> class ParticleLoop {

protected:
  /*
   * =================================================================
   */

  /**
   * create_kernel_arg is overloaded for each valid pair of access descriptor
   * and data structure which can be passesd to a loop.
   */
  /**
   *  Function to create the kernel argument for ParticleDat read access.
   */
  template <typename T>
  static inline void create_kernel_arg(const int cellx, const int layerx,
                                       T ***rhs,
                                       Access::ParticleDat::Read<T> &lhs) {
    lhs.layer = layerx;
    lhs.ptr = rhs[cellx];
  }
  /**
   *  Function to create the kernel argument for ParticleDat write access.
   */
  template <typename T>
  static inline void create_kernel_arg(const int cellx, const int layerx,
                                       T ***rhs,
                                       Access::ParticleDat::Write<T> &lhs) {
    lhs.layer = layerx;
    lhs.ptr = rhs[cellx];
  }
  /**
   *  Function to create the kernel argument for LocalArray read access.
   */
  template <typename T>
  static inline void create_kernel_arg(const int cellx, const int layerx,
                                       T const *rhs,
                                       Access::LocalArray::Read<T> &lhs) {
    lhs.ptr = rhs;
  }
  /**
   *  Function to create the kernel argument for LocalArray read access.
   */
  template <typename T>
  static inline void create_kernel_arg(const int cellx, const int layerx,
                                       T *rhs,
                                       Access::LocalArray::Add<T> &lhs) {
    lhs.ptr = rhs;
  }

  /*
   * -----------------------------------------------------------------
   */

  /**
   *  get_loop_arg is defined for each container and valid access type
   *  combination.
   */
  /**
   * Method to compute access to a particle dat (read)
   */
  template <typename T>
  static inline auto get_loop_arg(ParticleGroup *particle_group,
                                  sycl::handler &cgh, Access::Read<Sym<T>> a) {
    auto sym = a.obj;
    return particle_group->get_dat(sym)->cell_dat.device_ptr();
  }
  /**
   * Method to compute access to a particle dat (write)
   */
  template <typename T>
  static inline auto get_loop_arg(ParticleGroup *particle_group,
                                  sycl::handler &cgh, Access::Write<Sym<T>> a) {
    auto sym = a.obj;
    return particle_group->get_dat(sym)->cell_dat.device_ptr();
  }

  /**
   * Method to compute access to a LocalArray (read)
   */
  template <typename T>
  static inline auto get_loop_arg(ParticleGroup *particle_group,
                                  sycl::handler &cgh,
                                  Access::Read<LocalArray<T>> a) {
    return a.obj.impl_get_const();
  }
  /**
   * Method to compute access to a LocalArray (add)
   */
  template <typename T>
  static inline auto get_loop_arg(ParticleGroup *particle_group,
                                  sycl::handler &cgh,
                                  Access::Add<LocalArray<T>> a) {
    return a.obj.impl_get();
  }

  /*
   * =================================================================
   */

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

  /// Recursively assemble the outer loop arguments.
  template <size_t INDEX, size_t SIZE, typename PARAM>
  inline void create_loop_args_inner(ParticleGroup *pg, sycl::handler &cgh,
                                     PARAM &loop_args) {
    if constexpr (INDEX < SIZE) {
      Tuple::get<INDEX>(loop_args) =
          get_loop_arg(pg, cgh, std::get<INDEX>(this->args));
      create_loop_args_inner<INDEX + 1, SIZE>(pg, cgh, loop_args);
    }
  }
  inline void create_loop_args(sycl::handler &cgh,
                               loop_parameter_type &loop_args) {
    auto pg = this->particle_group.get();
    create_loop_args_inner<0, sizeof...(ARGS)>(pg, cgh, loop_args);
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
  std::unique_ptr<ParticleLoopIterationSet> iteration_set;
  std::string name;

  /**
   *  Launch the ParticleLoop and return. Pushes events onto an event stack
   *  that must be waited on for completion of the ParticleLoop.
   *
   *  @param event_stack EventStack to use for events.
   */
  inline void execute(EventStack &event_stack) {

    auto t0 = profile_timestamp();
    auto position_dat = this->particle_group->position_dat;
    auto d_npart_cell = position_dat->d_npart_cell;
    auto is = this->iteration_set->get();
    auto k_kernel = this->kernel;

    const int nbin = std::get<0>(is);
    this->particle_group->sycl_target->profile_map.inc(
        "ParticleLoop", "Init", 1, profile_elapsed(t0, profile_timestamp()));

    for (int binx = 0; binx < nbin; binx++) {
      sycl::nd_range<2> ndr = std::get<1>(is).at(binx);
      const size_t cell_offset = std::get<2>(is).at(binx);
      event_stack.push(this->particle_group->sycl_target->queue.submit(
          [&](sycl::handler &cgh) {
            loop_parameter_type loop_args;
            create_loop_args(cgh, loop_args);
            cgh.parallel_for<>(ndr, [=](sycl::nd_item<2> idx) {
              const size_t cellxs = idx.get_global_id(0) + cell_offset;
              const size_t layerxs = idx.get_global_id(1);
              const int cellx = static_cast<int>(cellxs);
              const int layerx = static_cast<int>(layerxs);
              if (layerx < d_npart_cell[cellx]) {
                kernel_parameter_type kernel_args;
                create_kernel_args(cellx, layerx, loop_args, kernel_args);
                Tuple::apply(k_kernel, kernel_args);
              }
            });
          }));
    }
  }

public:
  /// Disable (implicit) copies.
  ParticleLoop(const ParticleLoop &st) = delete;
  /// Disable (implicit) copies.
  ParticleLoop &operator=(ParticleLoop const &a) = delete;

  /**
   *  Create a ParticleLoop that executes a kernel for all particles in the
   * ParticleGroup.
   *
   *  @param name Identifier for particle loop.
   *  @param particle_group ParticleGroup to execute kernel for all particles.
   *  @param kernel Kernel to execute for all particles in the ParticleGroup.
   *  @param args The remaining arguments are arguments to be passed to the
   *              kernel. All arguments must be wrapped in an access descriptor
   * type.
   */
  ParticleLoop(const std::string name, ParticleGroupSharedPtr particle_group,
               KERNEL kernel, ARGS... args)
      : name(name), particle_group(particle_group), kernel(kernel) {
    this->unpack_args<0>(args...);

    const int ncell = this->particle_group->position_dat->ncell;
    auto h_npart_cell = this->particle_group->position_dat->h_npart_cell;
    this->iteration_set =
        std::make_unique<ParticleLoopIterationSet>(1, ncell, h_npart_cell);
  };

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
      : ParticleLoop("unnamed_kernel", particle_group, kernel, args...){};

  /**
   *  Execute the ParticleLoop and block until execution is complete.
   */
  inline void execute() {
    auto t0 = profile_timestamp();
    EventStack es;
    this->execute(es);
    es.wait();
    this->particle_group->sycl_target->profile_map.inc(
        "ParticleLoop", this->name, 1,
        profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO::Particles

#endif
