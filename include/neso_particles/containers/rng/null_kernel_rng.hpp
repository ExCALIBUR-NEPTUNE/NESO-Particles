#ifndef _NESO_PARTICLES_CONTAINERS_RNG_NULL_KERNEL_RNG_H_
#define _NESO_PARTICLES_CONTAINERS_RNG_NULL_KERNEL_RNG_H_

#include "kernel_rng.hpp"

namespace NESO::Particles {

/**
 * Null Kernel RNG that performs no operations.
 */
template <typename T> struct NullKernelRNGDevice {

  /**
   * @param row Unused, exists to match API.
   * @param col Unused, exists to match API.
   * @returns 0.
   */
  inline const T at([[maybe_unused]] const int &row,
                    [[maybe_unused]] const int &col) const {
    return static_cast<T>(0);
  }

  /**
   * @param[in] particle_index Unused, exists to match API.
   * @param[out] Unused, exists to match API.
   * @param[in, out] valid_sample Will be set to false.
   * @returns 0.
   */
  inline const T
  at([[maybe_unused]] const Access::LoopIndex::Read &particle_index,
     [[maybe_unused]] const int &component, bool *valid_sample) const {
    *valid_sample = false;
    return static_cast<T>(0);
  }
};

/**
 * KernelRNG implementation for RNG implementations which provide no RNG
 * samples. This type exists as an RNG implementation for downstream interfaces
 * which must provide a KernelRNG but the specific specialisation of the
 * downstream interface has no RNG requirements.
 */
template <typename T>
class NullKernelRNG : public KernelRNG<NullKernelRNGDevice<T>> {

public:
  NullKernelRNG() = default;
  virtual ~NullKernelRNG() = default;

  /**
   * No-op impl_get_const.
   *
   * @param global_info Unused, exists to match API.
   * @returns Null kernel type.
   */
  virtual inline Access::KernelRNG::Read<NullKernelRNGDevice<T>> impl_get_const(
      [[maybe_unused]] ParticleLoopImplementation::ParticleLoopGlobalInfo
          *global_info) override {
    return {};
  }

  /**
   * No-op impl_pre_loop_read.
   *
   * @param global_info Unused, exists to match API.
   */
  virtual inline void impl_pre_loop_read(
      [[maybe_unused]] ParticleLoopImplementation::ParticleLoopGlobalInfo
          *global_info) override {}

  /**
   * No-op impl_post_loop_read.
   * @param global_info Unused, exists to match API.
   */
  virtual inline void impl_post_loop_read(
      [[maybe_unused]] ParticleLoopImplementation::ParticleLoopGlobalInfo
          *global_info) override {}

  /**
   * @returns True.
   */
  virtual inline bool valid_internal_state() override { return true; };
};

/**
 * Helper function to create a NullKernelRNG.
 * @returns NullKernelRNG that can be passed as a ParticleLoop argument.
 */
template <typename T>
inline std::shared_ptr<NullKernelRNG<T>> null_kernel_rng() {
  return std::make_shared<NullKernelRNG<T>>();
}

} // namespace NESO::Particles

#endif
