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
 * The description of the iteration set to pass to the objects used in the
 * loop.
 */
struct ParticleLoopGlobalInfo {
  ParticleGroup *particle_group;
  INT *d_npart_cell_es;
  // The starting cell is only set for calls to create_loop_args.
  int starting_cell;
  int loop_type_int;
};

/**
 * The functions to run for each argument to the kernel post loop completion.
 */
/**
 * Default post loop execution function.
 */
template <typename T>
inline void post_loop(ParticleLoopGlobalInfo *global_info, T &arg) {}

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
