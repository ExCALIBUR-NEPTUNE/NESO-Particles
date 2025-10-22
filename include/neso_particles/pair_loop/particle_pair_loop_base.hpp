#ifndef _NESO_PARTICLES_PAIR_LOOP_PAIR_LOOP_BASE_HPP_
#define _NESO_PARTICLES_PAIR_LOOP_PAIR_LOOP_BASE_HPP_

#include "../loop/particle_loop_base.hpp"
#include <optional>

namespace NESO::Particles {

/**
 * Base type which all particle pair loops inherit from.
 */
class ParticlePairLoopBase {
protected:
public:
  virtual ~ParticlePairLoopBase() = default;
  ParticlePairLoopBase() = default;

  /**
   *  Execute the ParticlePairLoop and block until execution is complete. Must
   * be called collectively on the MPI communicator associated with the
   *  SYCLTarget this loop is over.
   *
   *  execute() Launches the ParticlePairLoop over all cells.
   *  execute(i) Launches the ParticlePairLoop over cell i.
   *  execute(i, i+4) Launches the ParticlePairLoop over cells i, i+1, i+2, i+3.
   *  Note cell_end itself is not visited.
   *
   *  @param cell_start Optional starting cell to launch the ParticlePairLoop
   * over.
   *  @param cell_end Optional ending cell to launch the ParticlePairLoop over.
   */
  virtual inline void
  execute(const std::optional<int> cell_start = std::nullopt,
          const std::optional<int> cell_end = std::nullopt) = 0;

  /**
   *  Launch the ParticlePairLoop and return. Must be called collectively over
   * the MPI communicator of the ParticleGroup. Loop execution is complete when
   *  the corresponding call to wait returns.
   *
   *  submit() Launches the ParticlePairLoop over all cells.
   *  submit(i) Launches the ParticlePairLoop over cell i.
   *  submit(i, i+4) Launches the ParticlePairLoop over cells i, i+1, i+2, i+3.
   *  Note cell_end itself is not visited.
   *
   *  @param cell_start Optional starting cell to launch the ParticlePairLoop
   * over.
   *  @param cell_end Optional ending cell to launch the ParticlePairLoop over.
   */
  virtual inline void
  submit(const std::optional<int> cell_start = std::nullopt,
         const std::optional<int> cell_end = std::nullopt) = 0;

  /**
   * Wait for loop execution to complete. On completion perform post-loop
   * actions. Must be called collectively on communicator.
   */
  virtual inline void wait() = 0;
};

using ParticlePairLoopBaseSharedPtr = std::shared_ptr<ParticlePairLoopBase>;

/**
 * Wrapper around a kernel for pair looping.
 */
template <typename KERNEL> struct ParticlePairLoopKernel {
  KERNEL kernel;
  ~ParticlePairLoopKernel() = default;
  ParticlePairLoopKernel() = default;
  ParticlePairLoopKernel(KERNEL kernel) : kernel(kernel) {}
};

/**
 * Wrap callable in ParticlePairLoopKernel if required.
 *
 * @param kernel Input kernel to wrap if required.
 * @returns ParticlePairLoopKernel wrapping kernel.
 */
template <typename KERNEL>
inline auto particle_pair_loop_kernel(KERNEL kernel) {
  return ParticlePairLoopKernel(kernel);
}

/**
 * Wrap callable in ParticlePairLoopKernel if required.
 *
 * @param kernel Input kernel to wrap if required.
 * @returns ParticlePairLoopKernel wrapping kernel.
 */
template <typename KERNEL>
inline auto particle_pair_loop_kernel(ParticlePairLoopKernel<KERNEL> kernel) {
  return kernel;
}

/**
 * Metafunction for getting kernel type.
 */
template <typename KERNEL> struct GetParticlePairLoopKernelType {
  using type = ParticlePairLoopKernel<KERNEL>;
};
/**
 * Metafunction for getting kernel type.
 */
template <typename KERNEL>
struct GetParticlePairLoopKernelType<ParticlePairLoopKernel<KERNEL>> {
  using type = ParticlePairLoopKernel<KERNEL>;
};

namespace Access {

/**
 * Wrapper for arguments that correspond to the first Particle{Sub}Group.
 */
template <typename OBJ> struct A {
  OBJ obj;
  A() = default;
  ~A() = default;
  A(OBJ obj) : obj(obj) {}
};

/**
 * Wrapper for arguments that correspond to the second Particle{Sub}Group.
 */
template <typename OBJ> struct B {
  OBJ obj;
  B() = default;
  ~B() = default;
  B(OBJ obj) : obj(obj) {}
};

/**
 * Metafunction for getting the unannotated loop type.
 */
template <typename ARG> struct StripPairGroupAnnotation {
  using type = ARG;
};
/**
 * Metafunction for getting the unannotated loop type.
 */
template <typename ARG> struct StripPairGroupAnnotation<A<ARG>> {
  using type = ARG;
};
/**
 * Metafunction for getting the unannotated loop type.
 */
template <typename ARG> struct StripPairGroupAnnotation<B<ARG>> {
  using type = ARG;
};

struct MaskA {};
struct MaskB {};
struct MaskUndefined {};

/**
 * Metafunction for determining if the arg is annotated or not.
 */
template <typename ARG> struct GetAnotateMask {
  using mask = MaskUndefined;
};
/**
 * Metafunction for determining if the arg is annotated or not.
 */
template <typename ARG> struct GetAnotateMask<A<ARG>> {
  using mask = MaskA;
};
/**
 * Metafunction for determining if the arg is annotated or not.
 */
template <typename ARG> struct GetAnotateMask<B<ARG>> {
  using mask = MaskB;
};

template <typename ARG> struct IsAnnotatedB {};
template <> struct IsAnnotatedB<MaskB &> {
  static constexpr bool value = true;
};
template <> struct IsAnnotatedB<MaskA &> {
  static constexpr bool value = false;
};
template <> struct IsAnnotatedB<MaskUndefined &> {
  static constexpr bool value = false;
};

/**
 * Helper function for retrieving the underlying ParticleLoop argument wrapped
 * in A,B.
 *
 * @param annotated_arg Argument potentially wrapped in Access::A or Access:B.
 * @returns Underlying argument wrapped in the access descriptor.
 */
template <typename ARG>
inline auto strip_pair_group_annotation(ARG &annotated_arg) {
  return annotated_arg;
}

/**
 * Helper function for retrieving the underlying ParticleLoop argument wrapped
 * in A,B.
 *
 * @param annotated_arg Argument potentially wrapped in Access::A or Access:B.
 * @returns Underlying argument wrapped in the access descriptor.
 */
template <typename ARG>
inline auto strip_pair_group_annotation(A<ARG> &annotated_arg) {
  return annotated_arg.obj;
}

/**
 * Helper function for retrieving the underlying ParticleLoop argument wrapped
 * in A,B.
 *
 * @param annotated_arg Argument potentially wrapped in Access::A or Access:B.
 * @returns Underlying argument wrapped in the access descriptor.
 */
template <typename ARG>
inline auto strip_pair_group_annotation(B<ARG> &annotated_arg) {
  return annotated_arg.obj;
}

} // namespace Access

} // namespace NESO::Particles

#endif
