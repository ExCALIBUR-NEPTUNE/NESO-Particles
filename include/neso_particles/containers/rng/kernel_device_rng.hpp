#ifndef _NESO_PARTICLES_KERNEL_DEVICE_RNG_H_
#define _NESO_PARTICLES_KERNEL_DEVICE_RNG_H_

#include "../../loop/particle_loop_base.hpp"
#include "../../loop/particle_loop_index.hpp"
#include "../../particle_group.hpp"

#include <functional>
#include <tuple>

namespace NESO::Particles {

template <typename T> struct KernelDeviceRNG;

namespace Access::KernelDeviceRNG {

/**
 * This is the kernel type for KernelDeviceRNG which is used for all
 * implementations which present RNG values to the kernel via an allocated
 * device buffer.
 */
template <typename T> struct Read {

  /**
   * Access the RNG data directly (advanced).
   *
   * @param row Row index to access.
   * @param col Column index to access.
   * @returns Constant reference to RNG data.
   */
  inline const T &at(const int row, const int col) const {
    // TODO
  }

  /**
   * Access the RNG data for this particle.
   *
   * @param particle_index Particle index to access.
   * @param component RNG component to access.
   * @returns Constant reference to RNG data.
   */
  inline const T &at(const Access::LoopIndex::Read &particle_index,
                     const int component) const {
    // TODO
  }
};

} // namespace Access::KernelDeviceRNG

namespace ParticleLoopImplementation {} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
