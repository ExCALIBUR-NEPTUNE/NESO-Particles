#ifndef _NESO_PARTICLES_HOST_ATOMIC_BLOCK_KERNEL_RNG_H_
#define _NESO_PARTICLES_HOST_ATOMIC_BLOCK_KERNEL_RNG_H_

#include "../../loop/particle_loop_utility.hpp"
#include "host_rng_common.hpp"
#include "kernel_rng.hpp"
#include <functional>
#include <tuple>

namespace NESO::Particles {

/**
 * ParticleLoop inner kernel type for accessing values via an incrementing
 * counter.
 */
template <typename T> struct AtomicBlockRNG {
  int buffer_size;
  int *RESTRICT counter;
  T const *RESTRICT d_ptr;
  inline T at(const Access::LoopIndex::Read &, const int, bool *valid_sample) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        element_atomic(this->counter[0]);
    const int index = element_atomic.fetch_add(1);
    bool valid_tmp = (index < buffer_size) && (0 <= index);

    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        poison_atomic(this->counter[1]);

    const int to_poison_int = static_cast<int>(!valid_tmp);
    const int already_poisoned_int = poison_atomic.fetch_max(to_poison_int);

    valid_tmp = valid_tmp && (!already_poisoned_int);
    *valid_sample = valid_tmp;
    return valid_tmp ? this->d_ptr[index] : static_cast<T>(0);
  }
};

/**
 * Class for RNG types where there is a host generation function that provides
 * values for a device buffer. Values are read from the device buffer by
 * atomically incrementing the counter and reading the value at the previous
 * counter value.
 */
template <typename T>
class HostAtomicBlockKernelRNG : public KernelRNG<AtomicBlockRNG<T>>,
                                 public BlockKernelRNGBase<T> {
protected:
  int internal_state;
  std::map<SYCLTargetSharedPtr, std::shared_ptr<BufferDevice<int>>> d_counters;
  std::map<SYCLTargetSharedPtr, int> num_values;

  inline void set_num_values(SYCLTargetSharedPtr sycl_target, const int value) {
    this->num_values[sycl_target] = value;
  }

  inline int get_num_values(SYCLTargetSharedPtr sycl_target) {
    if (!this->num_values.count(sycl_target)) {
      return 0;
    } else {
      return this->num_values.at(sycl_target);
    }
  }

  inline void create_counter(SYCLTargetSharedPtr sycl_target) {
    if (!this->d_counters.count(sycl_target)) {
      this->d_counters[sycl_target] =
          std::make_shared<BufferDevice<int>>(sycl_target, 2);
    }
  }

  inline int *get_counter_ptr(SYCLTargetSharedPtr sycl_target) {
    this->create_counter(sycl_target);
    return this->d_counters.at(sycl_target)->ptr;
  }

  inline int get_counter(SYCLTargetSharedPtr sycl_target, bool *is_poisoned) {
    int *ptr = this->get_counter_ptr(sycl_target);
    int count[2];
    sycl_target->queue.memcpy(&count, ptr, 2 * sizeof(int)).wait_and_throw();
    *is_poisoned = static_cast<bool>(count[1]);
    return count[0];
  }

  inline void reset_counter(SYCLTargetSharedPtr sycl_target) {
    int *ptr = this->get_counter_ptr(sycl_target);
    int count[2] = {0, 0};
    sycl_target->queue.memcpy(ptr, &count, 2 * sizeof(int)).wait_and_throw();
  }

public:
  int num_random_numbers_override;
  bool internal_state_is_valid;
  bool suppress_warnings;

  /**
   * Create the loop arguments for the RNG implementation.
   *
   * @param global_info Global information for the loop about to be executed.
   * @returns Pointer to the device data that contains the RNG data.
   */
  virtual inline Access::KernelRNG::Read<AtomicBlockRNG<T>> impl_get_const(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info)
      override {
    NESOASSERT((this->internal_state == 1) || (this->internal_state == 2),
               "Unexpected internal state.");
    this->internal_state = 2;
    if (this->num_components == 0) {
      return {0, nullptr, nullptr};
    } else {
      auto sycl_target = global_info->particle_group->sycl_target;
      const int buffer_size = this->get_num_values(sycl_target);
      return {buffer_size, this->get_counter_ptr(sycl_target),
              this->get_buffer_ptr(sycl_target)};
    }
  }

  /**
   * Executed by the loop pre execution.
   * @param global_info Global information for the loop which is to be executed.
   */
  virtual inline void impl_pre_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info)
      override {
    NESOASSERT(this->internal_state == 0,
               "HostKernelRNG Cannot be used within two loops which have "
               "overlapping execution.");
    this->internal_state = 1;

    if (this->num_components > 0) {
      const auto num_particles = get_loop_npart(global_info);
      auto sycl_target = global_info->particle_group->sycl_target;
      auto t0 = profile_timestamp();

      // Create num_particles * num_components random numbers from the RNG
      const std::size_t num_random_numbers =
          (this->num_random_numbers_override) > -1
              ? static_cast<std::size_t>(this->num_random_numbers_override)
              : static_cast<std::size_t>(num_particles) *
                    static_cast<std::size_t>(this->num_components);

      bool reallocated;
      auto d_ptr =
          this->allocate(sycl_target, num_random_numbers, &reallocated);

      if (reallocated) {
        draw_random_samples(sycl_target, this->generation_function, d_ptr,
                            num_random_numbers, this->block_size);
        this->set_num_values(sycl_target, num_random_numbers);
      } else {
        bool tmp_bool;
        const std::size_t current_count =
            static_cast<std::size_t>(this->get_counter(sycl_target, &tmp_bool));
        // Fill the block of numbers previously used.
        if (current_count > 0) {
          std::size_t num_required =
              std::min(num_random_numbers, current_count);
          draw_random_samples(sycl_target, this->generation_function, d_ptr,
                              num_required, this->block_size);
          if (num_required < current_count) {
            this->set_num_values(sycl_target, num_required);
          }
        }
        // Fill from the end of the previous required number of values to the
        // end of the new required number of values.
        const int previous_end = this->get_num_values(sycl_target);
        const int new_end = static_cast<int>(num_random_numbers);
        const int count_diff = new_end - previous_end;
        if (count_diff > 0) {
          draw_random_samples(
              sycl_target, this->generation_function, d_ptr + previous_end,
              static_cast<std::size_t>(count_diff), this->block_size);
          this->set_num_values(sycl_target, new_end);
        }
      }

      this->reset_counter(sycl_target);
      sycl_target->profile_map.inc("HostAtomicBlockKernelRNG",
                                   "impl_pre_loop_read", 1,
                                   profile_elapsed(t0, profile_timestamp()));
    }
  }

  /**
   * Executed by the loop post execution.
   * @param global_info Global information for the loop which completed.
   */
  virtual inline void impl_post_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info)
      override {
    this->internal_state = 0;
    auto sycl_target = global_info->particle_group->sycl_target;
    bool is_poisoned = false;
    const int read_count = this->get_counter(sycl_target, &is_poisoned);
    const int max_count = this->get_num_values(sycl_target);
    NESOWARN(((read_count <= max_count) && (!is_poisoned)) ||
                 this->suppress_warnings,
             "ParticleLoop attempted to read more RNG values than existed in "
             "the buffer. Buffer size is " +
                 std::to_string(max_count) +
                 " but the attempted read count was " +
                 std::to_string(read_count) + ".");
    if ((read_count > max_count) || is_poisoned) {
      this->internal_state_is_valid = false;
    }
  }

  /// The function pointer which returns samples when called.
  std::function<T()> generation_function;

  HostAtomicBlockKernelRNG() : BlockKernelRNGBase<T>() {}

  virtual ~HostAtomicBlockKernelRNG() = default;

  /**
   * Create a KernelRNG from a host function handle which returns values
   * of type T when called. For each loop invocation the implementation will
   * allocate a buffer equal to the number of particles in the loop times the
   * number of components per particle. The entries in this buffer are
   * allocated sequentially by atomically incrementing a counter for each call.
   *
   * @param func Host function handle which returns samples when called.
   * @param num_components Number of RNG values required per particle
   * (estimated maximum).
   * @param block_size Optional block size.
   */
  template <typename FUNC_TYPE>
  HostAtomicBlockKernelRNG(FUNC_TYPE func, const int num_components,
                           const int block_size = 8192)
      : BlockKernelRNGBase<T>(num_components, block_size), internal_state(0),
        num_random_numbers_override(-1), internal_state_is_valid(true),
        suppress_warnings(false), generation_function(func) {
    NESOASSERT(num_components >= 0, "Cannot have a RNG for " +
                                        std::to_string(num_components) +
                                        " components.");
  }

  /**
   * @returns True if no errors have been detected otherwise false.
   */
  virtual inline bool valid_internal_state() override {
    return this->internal_state_is_valid;
  }

  /**
   * Overrides the required number of sample values in the buffer.
   *
   * @param num_random_numbers Number of random numbers. Pass -1 to disable
   * override.
   */
  inline void set_num_random_numbers(const int num_random_numbers) {
    this->num_random_numbers_override = num_random_numbers;
  }
};

/**
 * Helper function to create a HostAtomicBlockKernelRNG around a host RNG
 * sampling function.
 *
 * @param func Host function which takes no arguments and returns a single
 * value of type T when called.
 * @param block_size Optional block size to sample RNG values and copy to the
 * device in.
 */
template <typename T, typename FUNC_TYPE>
inline std::shared_ptr<HostAtomicBlockKernelRNG<T>>
host_atomic_block_kernel_rng(FUNC_TYPE func, const int block_size = 8192) {
  return std::make_shared<HostAtomicBlockKernelRNG<T>>(func, block_size);
}

} // namespace NESO::Particles

#endif
