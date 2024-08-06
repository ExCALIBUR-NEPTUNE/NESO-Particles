#ifndef _NESO_PARTICLES_HOST_PER_PARTICLE_BLOCK_RNG_H_
#define _NESO_PARTICLES_HOST_PER_PARTICLE_BLOCK_RNG_H_

#include "../../loop/particle_loop_utility.hpp"
#include "host_rng_common.hpp"
#include "kernel_rng.hpp"
#include <functional>
#include <tuple>

namespace NESO::Particles {

/**
 * This is the kernel type for KernelRNG which is used for all implementations
 * which present RNG values to the kernel via an allocated device buffer.
 */
template <typename T> struct PerParticleBlockRNG {

  /**
   * Stride for matrix access. We assume the matrix is store column wise and a
   * particle accesses a row. This stride is essentially the number of rows
   * (number of particles).
   */
  int stride;

  /**
   * Pointer to the device buffer for all the RNG values.
   */
  T const *RESTRICT d_ptr;

  /**
   * Access the RNG data directly (advanced).
   *
   * @param row Row index to access.
   * @param col Column index to access.
   * @returns Constant reference to RNG data.
   */
  inline const T &at(const int row, const int col) const {
    return d_ptr[col * stride + row];
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
    const auto index = particle_index.get_loop_linear_index();
    return at(index, component);
  }
};

/**
 * KernelRNG implementation for RNG implementations which call a host function
 * to sample a value.
 */
template <typename T>
class HostPerParticleBlockRNG : public KernelRNG<PerParticleBlockRNG<T>>,
                                public BlockKernelRNGBase<T> {
protected:
  int internal_state;

public:
  /// The function pointer which returns samples when called.
  std::function<T()> generation_function;

  /**
   * Create a KernelRNG from a host function handle which returns values of
   * type T when called.
   *
   * @param func Host function handle which returns samples when called.
   * @param num_components Number of RNG values required per particle.
   * @param block_size Optional block size.
   */
  template <typename FUNC_TYPE>
  HostPerParticleBlockRNG(FUNC_TYPE func, const int num_components,
                          const int block_size = 8192)
      : BlockKernelRNGBase<T>(num_components, block_size),
        generation_function(func), internal_state(0) {
    NESOASSERT(num_components >= 0, "Cannot have a RNG for " +
                                        std::to_string(num_components) +
                                        " components.");
  }

  /**
   * Create the loop arguments for the RNG implementation.
   *
   * @param global_info Global information for the loop about to be executed.
   * @returns Pointer to the device data that contains the RNG data.
   */
  virtual inline Access::KernelRNG::Read<PerParticleBlockRNG<T>> impl_get_const(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info)
      override {
    NESOASSERT((this->internal_state == 1) || (this->internal_state == 2),
               "Unexpected internal state.");
    this->internal_state = 2;
    if (this->num_components == 0) {
      return {0, nullptr};
    } else {
      const auto num_particles = get_loop_npart(global_info);
      auto sycl_target = global_info->particle_group->sycl_target;
      return {num_particles, this->get_buffer_ptr(sycl_target)};
    }
  }

  /**
   * Create the loop arguments for the RNG implementation.
   *
   * @param global_info Global information for the loop about to be executed.
   * @returns Pointer to the device data that contains the RNG data.
   */
  virtual inline void impl_pre_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info)
      override {
    NESOASSERT(
        this->internal_state == 0,
        "HostPerParticleBlockRNG Cannot be used within two loops which have "
        "overlapping execution.");
    this->internal_state = 1;
    if (this->num_components > 0) {
      const auto num_particles = get_loop_npart(global_info);

      // Allocate space
      auto sycl_target = global_info->particle_group->sycl_target;
      auto t0 = profile_timestamp();
      auto d_ptr = this->allocate(sycl_target, num_particles);
      // Create num_particles * num_components random numbers from the RNG
      const std::size_t num_random_numbers =
          static_cast<std::size_t>(num_particles) *
          static_cast<std::size_t>(this->num_components);

      draw_random_samples(sycl_target, this->generation_function, d_ptr,
                          num_random_numbers, this->block_size);

      sycl_target->profile_map.inc("HostPerParticleBlockRNG",
                                   "impl_pre_loop_read", 1,
                                   profile_elapsed(t0, profile_timestamp()));
    }
  }

  /**
   * Ran executed by the loop post execution.
   * @param global_info Global information for the loop which completed.
   */
  virtual inline void impl_post_loop_read(
      [[maybe_unused]] ParticleLoopImplementation::ParticleLoopGlobalInfo
          *global_info) override {
    NESOASSERT(this->internal_state == 2,
               "HostPerParticleBlockRNG Unexpected state, post loop called but "
               "internal "
               "state does not expect a loop to be running.");
    this->internal_state = 0;
  }

  virtual inline bool valid_internal_state() override { return true; };
};

/**
 * Helper function to create a KernelRNG around a host RNG sampling function.
 *
 * @param func Host function which takes no arguments and returns a single
 * value of type T when called.
 * @param num_components Number of samples required per particle in the kernel.
 * @param block_size Optional block size to sample RNG values and copy to the
 * device in.
 */
template <typename T, typename FUNC_TYPE>
inline std::shared_ptr<HostPerParticleBlockRNG<T>>
host_per_particle_block_rng(FUNC_TYPE func, const int num_components,
                            const int block_size = 8192) {
  return std::make_shared<HostPerParticleBlockRNG<T>>(func, num_components,
                                                      block_size);
}

} // namespace NESO::Particles

#endif
