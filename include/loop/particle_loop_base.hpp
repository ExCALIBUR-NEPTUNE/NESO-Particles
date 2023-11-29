#ifndef _NESO_PARTICLES_PARTICLE_LOOP_BASE_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_BASE_H_

#include "../compute_target.hpp"
#include "access_descriptors.hpp"
#include <memory>
#include <optional>

namespace NESO::Particles {
class ParticleGroup;

namespace ParticleLoopImplementation {

/**
 * The LoopParameter types define the type collected from a data structure
 * prior to calling the loop. The loop is responsible for computing the kernel
 * argument from the loop argument. e.g. in the ParticleDat case the
 * LoopParameter is the pointer type that points to the data for all cells,
 * layers and components.
 */
template <typename T> struct LoopParameter { using type = void *; };

/**
 * The KernelParameter types define the types passed to the kernel for each
 * data structure type for each access descriptor.
 */
template <typename T> struct KernelParameter { using type = void; };

/**
 * The functions to run for each argument to the kernel post loop completion.
 */
/**
 * Default post loop execution function.
 */
template <typename T>
inline void post_loop(ParticleGroup *particle_group, T &arg) {}

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

/**
 * The type to pass to a ParticleLoop to read the ParticleLoop loop index in a
 * kernel.
 */
struct ParticleLoopIndex {};

/**
 * Defines the access type for the cell, layer indexing.
 */
namespace Access::LoopIndex {
/**
 * ParticleLoop index containing the cell and layer.
 */
struct Read {
  /// The cell containing the particle.
  INT cell;
  /// The layer of the particle.
  INT layer;
};

} // namespace Access::LoopIndex

namespace ParticleLoopImplementation {

/**
 *  KernelParameter type for read-only access to a ParticleLoopIndex.
 */
template <> struct KernelParameter<Access::Read<ParticleLoopIndex>> {
  using type = Access::LoopIndex::Read;
};
/**
 *  Loop parameter for read access of a ParticleLoopIndex.
 */
template <> struct LoopParameter<Access::Read<ParticleLoopIndex>> {
  using type = void *;
};
/**
 * Method to compute access to a ParticleLoopIndex (read)
 */
static inline void *create_loop_arg(ParticleGroup *particle_group,
                                    sycl::handler &cgh,
                                    Access::Read<ParticleLoopIndex *> &a) {
  return nullptr;
}
/**
 *  Function to create the kernel argument for ParticleLoopIndex read access.
 */
inline void create_kernel_arg(const int cellx, const int layerx, void *,
                              Access::LoopIndex::Read &lhs) {
  lhs.cell = cellx;
  lhs.layer = layerx;
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles
#endif
