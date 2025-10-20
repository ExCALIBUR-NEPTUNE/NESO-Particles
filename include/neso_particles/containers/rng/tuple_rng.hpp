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

  template <std::size_t INDEX> inline auto &get() {
    return Tuple::get<INDEX>(this->rngs);
  }
};

/**
 * Helper function that calls Access::TupleRNG::get in a way that avoids using
 * ".template".
 *
 * @param tuple_rng Access::TupleRNG::Read instance to access.
 * @returns The reference returned by Access::TupleRNG::Read::get.
 */
template <std::size_t INDEX, typename... KERNELRNGS>
auto &get(Read<KERNELRNGS...> &tuple_rng) {
  return tuple_rng.template get<INDEX>();
}

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
 * Container type which holds multiple RNG types which inherit from KernelRNG or
 * are also TupleRNG instances.
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
   * Constructor to create a tuple of KernelRNG/TupleRNG instances.
   *
   * @param args Shared pointers to instances of KernelRNG descendent types and
   * TupleRNG instances.
   */
  TupleRNG(RNGPTRS... args) { this->rng_ptrs = Tuple::to_tuple(args...); }

protected:
  template <std::size_t RX>
  inline void impl_get_const_inner(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      KernelType *kernel_type) {

    Tuple::get<RX>(kernel_type->rngs) =
        Tuple::get<RX>(this->rng_ptrs)->impl_get_const(global_info);
    if constexpr ((RX + 1) < (sizeof...(RNGPTRS))) {
      this->impl_get_const_inner<RX + 1>(global_info, kernel_type);
    }
  }

public:
  /**
   * Called by ParticleLoop to create the loop arguments.
   *
   * @param global_info Global information for the loop which is to be executed.
   */
  inline KernelType impl_get_const(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {

    KernelType r;
    this->impl_get_const_inner<0>(global_info, &r);
    return r;
  }

  /**
   * Executed by the loop pre execution.
   * @param global_info Global information for the loop which is to be executed.
   */
  inline void impl_pre_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {

    auto lambda_apply_pre_loop = [&](auto &rng) {
      rng->impl_pre_loop_read(global_info);
    };
    Tuple::apply([&](auto... ptrs) { (lambda_apply_pre_loop(ptrs), ...); },
                 this->rng_ptrs);
  }

  /**
   * Executed by the loop post execution.
   * @param global_info Global information for the loop which completed.
   */
  void impl_post_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {

    auto lambda_apply_post_loop = [&](auto &rng) {
      rng->impl_post_loop_read(global_info);
    };
    Tuple::apply([&](auto... ptrs) { (lambda_apply_post_loop(ptrs), ...); },
                 this->rng_ptrs);
  }
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
  return a.obj->impl_get_const(global_info);
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

/**
 * The function called before the ParticleLoop is executed.
 */
template <typename... T>
inline void pre_loop(ParticleLoopGlobalInfo *global_info,
                     Access::Read<TupleRNG<T...> *> &arg) {
  arg.obj->impl_pre_loop_read(global_info);
}

/**
 * The function called after the ParticleLoop is executed.
 */
template <typename... T>
inline void post_loop(ParticleLoopGlobalInfo *global_info,
                      Access::Read<TupleRNG<T...> *> &arg) {
  arg.obj->impl_post_loop_read(global_info);
}

} // namespace ParticleLoopImplementation

/**
 * Helper function to create a tuple of KernelRNG/TupleRNG instances.
 *
 * @param args Shared pointers to instances of KernelRNG descendent types and
 * TupleRNG instances.
 * @returns New TupleRNG instance which can be passed to a particle loop with
 * read access.
 */
template <typename... RNGPTRS> auto tuple_rng(RNGPTRS... rng_ptrs) {
  return std::make_shared<TupleRNG<RNGPTRS...>>(rng_ptrs...);
}

} // namespace NESO::Particles

#endif
