#ifndef _NESO_PARTICLES_CONTAINERS_PARTICLE_MASK_HPP_
#define _NESO_PARTICLES_CONTAINERS_PARTICLE_MASK_HPP_

#include <cstdint>
#include <memory>
#include <optional>

#include "../compute_target.hpp"
#include "../device_buffers.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../loop/particle_loop_index.hpp"
#include "../particle_group.hpp"
#include "mask_array.hpp"

namespace NESO::Particles {

class ParticleMask;

namespace Access::ParticleMask {

struct Read {
  Access::MaskArray::Read mask_array;

  /**
   * @returns The current mask value for the particle.
   *
   * @param index ParticleLoopIndex of particle to get mask for.
   */
  inline bool get(const Access::LoopIndex::Read &index) const {
    return this->mask_array.get(index.get_local_linear_index(), 0);
  }
};

struct Write {
  std::size_t entry_index{0};
  Access::MaskArray::Write mask_array;

  /**
   * @returns The current mask value for the particle.
   *
   * @param index ParticleLoopIndex of particle to get mask for.
   */
  inline bool get(const Access::LoopIndex::Read &index) const {
    return this->mask_array.get(index.get_local_linear_index(), 0);
  }

  /**
   * Set the mask to on.
   *
   * @param index ParticleLoopIndex of particle to set mask for.
   */
  inline void set_on(const Access::LoopIndex::Read &index) {
    this->mask_array.set(index.get_local_linear_index(), 0, true);
  }

  /**
   * Set the mask to off.
   *
   * @param index ParticleLoopIndex of particle to set mask for.
   */
  inline void set_off(const Access::LoopIndex::Read &index) {
    this->mask_array.set(index.get_local_linear_index(), 0, false);
  }

  /**
   * Set the mask.
   *
   * @param index ParticleLoopIndex of particle to set mask for.
   * @param value Mask value to set..
   */
  inline void set(const Access::LoopIndex::Read &index, const bool value) {
    this->mask_array.set(index.get_local_linear_index(), 0, value);
  }
};

} // namespace Access::ParticleMask

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a ParticleMask.
 */
template <> struct LoopParameter<Access::Read<ParticleMask>> {
  using type = Access::MaskArray::Read;
};

/**
 *  Loop parameter for write access of a ParticleMask.
 */
template <> struct LoopParameter<Access::Write<ParticleMask>> {
  using type = Access::MaskArray::Write;
};

/**
 *  KernelParameter type for Read access to a ParticleMask.
 */
template <> struct KernelParameter<Access::Read<ParticleMask>> {
  using type = Access::ParticleMask::Read;
};

/**
 *  KernelParameter type for write access to a ParticleMask.
 */
template <> struct KernelParameter<Access::Write<ParticleMask>> {
  using type = Access::ParticleMask::Write;
};

/**
 * Method to compute access to a MaskArray (read)
 */
Access::MaskArray::Read
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ParticleMask *> &a);

/**
 * Method to compute access to a MaskArray (write)
 */
Access::MaskArray::Write
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ParticleMask *> &a);

/**
 *  Function to create the kernel argument for ParticleMask read access.
 */
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::MaskArray::Read &rhs,
                  Access::ParticleMask::Read &lhs) {
  lhs.mask_array = rhs;
}

/**
 *  Function to create the kernel argument for ParticleMask write access.
 */
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::MaskArray::Write &rhs,
                  Access::ParticleMask::Write &lhs) {
  lhs.mask_array = rhs;
}

} // namespace ParticleLoopImplementation

/**
 * Container to store a single mask per particle which is accessible from
 * ParticleLoop.
 */
class ParticleMask : public MaskArray {
protected:
public:
  /**
   * Create a new ParticleMask on a compute device.
   *
   * @param sycl_target Compute device to create mask on.
   */
  ParticleMask(SYCLTargetSharedPtr sycl_target);

  /**
   * Reset the container to be able to store masks for all particles within the
   * passed ParticleGroup.
   *
   * @param particle_group ParticleGroup to reset size for.
   */
  void reset(ParticleGroupSharedPtr particle_group);

  /**
   * Set the masks from a ParticleDat Sym and component. Calls reset using the
   * ParticleGroup. Calls reset before setting.
   *
   * @param particle_group ParticleGroup to set from.
   * @param sym Sym for ParticleDat to set from.
   * @param component ParticleDat component to set from.
   */
  void set(ParticleGroupSharedPtr particle_group, Sym<INT> sym,
           const int component);

  /**
   * Set all the masks to a specified value. Calls reset before setting.
   *
   * @param particle_group ParticleGroup to set from.
   * @param value Value to set.
   */
  void set(ParticleGroupSharedPtr particle_group, const bool value);

  /**
   * Store the masks in a ParticleDat Sym and component.
   *
   * @param particle_group ParticleGroup to set into.
   * @param sym Sym for ParticleDat to set int.
   * @param component ParticleDat component to set into.
   */
  void get(ParticleGroupSharedPtr particle_group, Sym<INT> sym,
           const int component);
};

using ParticleMaskSharedPtr = std::shared_ptr<ParticleMask>;

} // namespace NESO::Particles

#endif
