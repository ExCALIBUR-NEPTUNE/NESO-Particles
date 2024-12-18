#ifndef _NESO_PARTICLES_PARTICLE_LOOP_BASE_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_BASE_H_

#include "../compute_target.hpp"
#include "../device_buffers.hpp"
#include "access_descriptors.hpp"
#include <memory>
#include <optional>
#include <type_traits>

namespace NESO::Particles {
class ParticleGroup;
class ParticleSubGroup;

namespace ParticleLoopImplementation {

/**
 * The description of the iteration set to pass to the objects used in the
 * loop.
 */
struct ParticleLoopGlobalInfo {
  // Underlying @ref ParticleGroup that created the iteration set.
  ParticleGroup *particle_group;
  // If the iteration set is actually a @ref ParticleSubGroup then this is a
  // pointer to the sub group. Otherwise this member is a nullptr.
  ParticleSubGroup *particle_sub_group;
  INT *d_npart_cell_es;
  INT *d_npart_cell_es_lb;
  // The starting cell is only set for calls to create_loop_args.
  int starting_cell;
  // Last cell plus one. Only set for calls to create_loop_args.
  int bounding_cell;
  int loop_type_int;
  std::size_t local_size;
};

/**
 * The LoopParameter types define the type collected from a data structure
 * prior to calling the loop. The loop is responsible for computing the kernel
 * argument from the loop argument. e.g. in the ParticleDat case the
 * LoopParameter is the pointer type that points to the data for all cells,
 * layers and components.
 */
// template <typename T> struct LoopParameter { using type = void *; };
template <typename T, typename U = std::true_type> struct LoopParameter {
  using type = void *;
};

/**
 * The KernelParameter types define the types passed to the kernel for each
 * data structure type for each access descriptor.
 */
// template <typename T> struct KernelParameter { using type = void; };
template <typename T, typename U = std::true_type>
struct KernelParameter; // { using type = void; };

/**
 * The description of the iteration index to pas to objects used in the loop.
 */
struct ParticleLoopIteration {
  /// The local index in the sycl nd range.
  std::size_t local_sycl_index;
  /// The cell the particle resides in.
  int cellx;
  /// The layer (row) the particle resides in.
  int layerx;
  /// The layer of the particle in the particle loop iteration set. e.g. not
  /// all looping types visit all the particles in a cell. layerx_loop is a
  /// linear contiguous index for the particles which are visited in each cell.
  int loop_layerx;
};

/**
 * The functions to run for each argument to the kernel post loop completion.
 */
/**
 * Default pre loop execution function.
 */
template <typename T>
inline void pre_loop([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                     [[maybe_unused]] T &arg) {}
/**
 * Default post loop execution function.
 */
template <typename T>
inline void post_loop([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                      [[maybe_unused]] T &arg) {}
/**
 * Default function to determine how much local space a type needs per particle.
 */
template <typename T>
inline std::size_t get_required_local_num_bytes([[maybe_unused]] T &arg) {
  return 0;
}

/**
 * Extract the kernel from an object for a ParticleLoop.
 * @param kernel Device copyable callable type to use as kernel.
 * @returns kernel.
 */
template <typename T> inline T &get_kernel(T &kernel) { return kernel; }

/**
 * Extract the number of bytes per kernel invocation from the kernel. No-op
 * implementation for when kernel is a generic callable.
 *
 * @param kernel Device copyable callable type to use as kernel.
 * @returns 0.
 */
template <typename T>
inline std::size_t get_kernel_num_bytes([[maybe_unused]] T &kernel) {
  return 0;
}

/**
 * Extract the number of flops per kernel invocation from the kernel. No-op
 * implementation for when kernel is a generic callable.
 *
 * @param kernel Device copyable callable type to use as kernel.
 * @returns 0.
 */
template <typename T>
inline std::size_t get_kernel_num_flops([[maybe_unused]] T &kernel) {
  return 0;
}

} // namespace ParticleLoopImplementation

/**
 * Abstract base class for ParticleLoop such that the templated ParticleLoop
 * can be cast to a base type for storage.
 */
class ParticleLoopBase {
public:
  /**
   *  Execute the particle loop and block until completion. Must be called
   *  Collectively on the communicator.
   */
  virtual inline void execute(const std::optional<int> cell = std::nullopt) = 0;

  /**
   *  Launch the ParticleLoop and return. Must be called collectively over the
   *  MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   */
  virtual inline void submit(const std::optional<int> cell = std::nullopt) = 0;

  /**
   * Wait for loop execution to complete. On completion perform post-loop
   * actions. Must be called collectively on communicator.
   */
  virtual inline void wait() = 0;
};

typedef std::shared_ptr<ParticleLoopBase> ParticleLoopSharedPtr;

} // namespace NESO::Particles
#endif
