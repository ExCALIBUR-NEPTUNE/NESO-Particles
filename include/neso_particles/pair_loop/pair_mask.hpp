#ifndef _NESO_PARTICLES_PAIR_LOOP_PAIR_MASK_HPP_
#define _NESO_PARTICLES_PAIR_LOOP_PAIR_MASK_HPP_

#include "../containers/mask_array.hpp"

namespace NESO::Particles {

class PairMask;

namespace Access::PairMask {

struct Write {
  std::size_t entry_index{0};
  MaskArrayDevice mask_array;

  /**
   * @returns The current mask value for the pair.
   */
  inline bool get() const { return this->mask_array.get(entry_index, 0); }

  /**
   * Enable this pair. Note that this method is only functionally useful if
   * for some reason the pair was disabled within the same kernel.
   */
  inline void set_on() { this->mask_array.set(entry_index, 0, true); }

  /**
   * Mask off this pair.
   */
  inline void set_off() { this->mask_array.set(entry_index, 0, false); }
};
} // namespace Access::PairMask

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for write access of a PairMask.
 */
template <> struct LoopParameter<Access::Write<PairMask>> {
  using type = MaskArrayDevice;
};

/**
 *  KernelParameter type for write access to a PairMask.
 */
template <> struct KernelParameter<Access::Write<PairMask>> {
  using type = Access::PairMask::Write;
};

} // namespace ParticleLoopImplementation

namespace ParticlePairLoopImplementation {

/**
 * Function to create the kernel argument for PairMask write
 * access in a pair loop.
 */
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    MaskArrayDevice &rhs, Access::PairMask::Write &lhs) {
  lhs = {static_cast<std::size_t>(iteration.pair_index), rhs};
}

} // namespace ParticlePairLoopImplementation

class PairMask : public MaskArray {
protected:
public:
  PairMask() = default;
  virtual ~PairMask() = default;

  /**
   * Create a pair mask on a given compute device with a set number of bits per
   * entry.
   *
   * @param sycl_target Compute device.
   */
  PairMask(SYCLTargetSharedPtr sycl_target);
};

using PairMaskSharedPtr = std::shared_ptr<PairMask>;

namespace ParticleLoopImplementation {

/**
 * Method to compute access to a PairMask (write)
 */
inline MaskArrayDevice
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<PairMask *> &a) {
  auto tmp = a.obj->get_device();
  return tmp;
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
