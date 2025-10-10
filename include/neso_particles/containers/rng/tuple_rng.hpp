#ifndef _NESO_PARTICLES_CONTAINERS_RNG_TUPLE_RNG_H_
#define _NESO_PARTICLES_CONTAINERS_RNG_TUPLE_RNG_H_

#include <memory>

#include "../../containers/tuple.hpp"
#include "kernel_rng.hpp"

namespace NESO::Particles {

// Forward declaration.
template <typename... RNGPTRS> class TupleRNG;

namespace Access::TupleRNG {

/**
 * This is the kernel type for TupleRNG.
 */
template <typename... KERNELRNGS> struct Read {
  Tuple::Tuple<KERNELRNGS...> rngs;
};

} // namespace Access::TupleRNG

namespace Private {

// base template
template <typename... T> struct GetTupleRNGDeviceType {};
// template specialisation for TupleRNG
template <typename... RNGPTRS>
struct GetTupleRNGDeviceType<TupleRNG<RNGPTRS...>> {
  using type = typename TupleRNG<RNGPTRS...>::KernelType;
};
// template specialisation for KernelRNG
template <typename T> struct GetTupleRNGDeviceType<T> {
  using type = typename ParticleLoopImplementation::KernelParameter<
      Access::Read<T>>::type;
};
// Interface metafunction to recursively call GetTupleRNGDeviceType
template <typename... RNGPTRS> struct GetTupleRNGDeviceTypes {
  using type =
      Access::TupleRNG::Read<typename GetTupleRNGDeviceType<RNGPTRS>::type...>;
};
} // namespace Private

/**
 * TODO
 */
template <typename... RNGPTRS> class TupleRNG {
protected:
  Tuple::Tuple<RNGPTRS...> rng_ptrs;

public:
  using KernelType = typename Private::GetTupleRNGDeviceTypes<
      typename RNGPTRS::element_type...>::type;

  ~TupleRNG() = default;
  TupleRNG() = default;

  /**
   * TODO
   */
  TupleRNG(RNGPTRS... args) { this->rng_ptrs = Tuple::to_tuple(args...); }
};

namespace ParticleLoopImplementation {

/**
 * LoopParameter type for a TupleRNG.
 */
template <typename... T> struct LoopParameter<Access::Read<TupleRNG<T...>>> {
  using type = typename TupleRNG<T...>::KernelType;
};
/**
 * KernelParameter type for a TupleRNG.
 */
template <typename... T> struct KernelParameter<Access::Read<TupleRNG<T...>>> {
  using type = typename TupleRNG<T...>::KernelType;
};

/**
 * Create the loop argument for a TupleRNG.
 */
template <typename... T>
inline typename TupleRNG<T...>::KernelType
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<TupleRNG<T...> *> &a) {

  // TODO
  typename TupleRNG<T...>::KernelType b;

  return b;
}

/**
 * Create the kernel argument for a TupleRNG.
 */
template <typename... T>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::TupleRNG::Read<T...> &rhs,
                  Access::TupleRNG::Read<T...> &lhs) {
  lhs = rhs;
}

} // namespace ParticleLoopImplementation

/**
 * TODO
 */
template <typename... RNGPTRS> auto tuple_rng(RNGPTRS... rng_ptrs) {
  return std::make_shared<TupleRNG<RNGPTRS...>>(rng_ptrs...);
}

} // namespace NESO::Particles

#endif
