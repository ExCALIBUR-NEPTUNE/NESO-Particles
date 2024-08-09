#ifndef _NESO_PARTICLES_HOST_KERNEL_RNG_H_
#define _NESO_PARTICLES_HOST_KERNEL_RNG_H_

#include "kernel_rng.hpp"

#include "../../particle_sub_group.hpp"
#include <functional>
#include <tuple>

namespace NESO::Particles {

/**
 * KernelRNG implementation for RNG implementations which call a host function
 * to sample a value.
 */
template <typename T> class HostKernelRNG : public KernelRNG<T> {
protected:
  int internal_state;
  std::map<SYCLTargetSharedPtr, std::shared_ptr<BufferDevice<T>>> d_buffers;

  inline T *allocate(SYCLTargetSharedPtr sycl_target, const int nrow) {
    if (nrow <= 0) {
      return nullptr;
    }
    const std::size_t required_size =
        static_cast<std::size_t>(nrow) *
        static_cast<std::size_t>(this->num_components);

    if (!this->d_buffers.count(sycl_target)) {
      this->d_buffers[sycl_target] =
          std::make_unique<BufferDevice<T>>(sycl_target, required_size);
    } else {
      this->d_buffers.at(sycl_target)->realloc_no_copy(required_size, 1.2);
    }

    return this->d_buffers.at(sycl_target)->ptr;
  }

  inline int
  get_npart(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) {
    const int cell_start = global_info->starting_cell;
    const int cell_end = global_info->bounding_cell;

    int num_particles;
    if ((cell_end - cell_start) == 1) {
      // Single cell looping case
      // Allocate for all the particles in the cell.
      if (global_info->particle_sub_group != nullptr) {
        num_particles =
            global_info->particle_sub_group->get_npart_cell(cell_start);
      } else {
        num_particles = global_info->particle_group->get_npart_cell(cell_start);
      }
    } else {
      // Whole domain looping case
      if (global_info->particle_sub_group != nullptr) {
        num_particles = global_info->particle_sub_group->get_npart_local();
      } else {
        num_particles = global_info->particle_group->get_npart_local();
      }
    }
    return num_particles;
  }

public:
  /// The number of RNG values required per particle.
  int num_components;
  /// RNG values are sampled and copied to the device in this block size.
  int block_size;
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
  HostKernelRNG(FUNC_TYPE func, const int num_components,
                const int block_size = 8192)
      : generation_function(func), num_components(num_components),
        internal_state(0), block_size(block_size) {
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
  virtual inline std::tuple<int, T *> impl_get_const(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info)
      override {
    NESOASSERT((this->internal_state == 1) || (this->internal_state == 2),
               "Unexpected internal state.");
    this->internal_state = 2;
    if (this->num_components == 0) {
      return {0, nullptr};
    } else {
      const auto num_particles = this->get_npart(global_info);
      auto sycl_target = global_info->particle_group->sycl_target;
      return {num_particles, this->d_buffers.at(sycl_target)->ptr};
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
    NESOASSERT(this->internal_state == 0,
               "HostKernelRNG Cannot be used within two loops which have "
               "overlapping execution.");
    this->internal_state = 1;
    if (this->num_components > 0) {
      const auto num_particles = this->get_npart(global_info);

      // Allocate space
      auto sycl_target = global_info->particle_group->sycl_target;
      auto t0 = profile_timestamp();
      auto d_ptr = this->allocate(sycl_target, num_particles);
      auto d_ptr_start = d_ptr;

      // Create num_particles * num_components random numbers from the RNG
      const std::size_t num_random_numbers =
          static_cast<std::size_t>(num_particles) *
          static_cast<std::size_t>(num_components);

      // Create the random number in blocks and copy to device blockwise.
      std::vector<T> block0(this->block_size);
      std::vector<T> block1(this->block_size);

      T *ptr_tmp;
      T *ptr_current = block0.data();
      T *ptr_next = block1.data();
      std::size_t num_numbers_moved = 0;

      sycl::event e;
      while (num_numbers_moved < num_random_numbers) {

        // Create a block of samples
        const std::size_t num_to_memcpy =
            std::min(static_cast<std::size_t>(this->block_size),
                     num_random_numbers - num_numbers_moved);
        for (std::size_t ix = 0; ix < num_to_memcpy; ix++) {
          ptr_current[ix] = this->generation_function();
        }

        // Wait until the previous block finished copying before starting this
        // copy
        e.wait_and_throw();
        e = sycl_target->queue.memcpy(d_ptr, ptr_current,
                                      num_to_memcpy * sizeof(T));
        d_ptr += num_to_memcpy;
        num_numbers_moved += num_to_memcpy;

        // swap ptr_current and ptr_next such that the new samples are written
        // into ptr_next whilst ptr_current is being copied to the device.
        ptr_tmp = ptr_current;
        ptr_current = ptr_next;
        ptr_next = ptr_tmp;
      }

      e.wait_and_throw();
      sycl_target->profile_map.inc("HostKernelRNG", "impl_pre_loop_read", 1,
                                   profile_elapsed(t0, profile_timestamp()));

      NESOASSERT(num_numbers_moved == num_random_numbers,
                 "Failed to copy the correct number of random numbers");
      NESOASSERT(d_ptr == d_ptr_start + num_random_numbers,
                 "Failed to copy the correct number of random numbers (pointer "
                 "arithmetic)");
    }
  }

  /**
   * Ran executed by the loop post execution.
   * @param global_info Global information for the loop which completed.
   */
  virtual inline void impl_post_loop_read(
      [[maybe_unused]] ParticleLoopImplementation::ParticleLoopGlobalInfo
          *global_info) override {
    this->internal_state = 0;
  }
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
inline std::shared_ptr<KernelRNG<T>>
host_kernel_rng(FUNC_TYPE func, const int num_components,
                const int block_size = 8192) {
  return std::dynamic_pointer_cast<KernelRNG<T>>(
      std::make_shared<HostKernelRNG<T>>(func, num_components, block_size));
}

} // namespace NESO::Particles

#endif
