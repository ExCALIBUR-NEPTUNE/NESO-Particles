#ifndef _NESO_PARTICLES_KERNEL_RNG_H_
#define _NESO_PARTICLES_KERNEL_RNG_H_

#include "../loop/particle_loop_base.hpp"
#include "../loop/particle_loop_index.hpp"
#include "../particle_group.hpp"

#include <functional>
#include <tuple>

namespace NESO::Particles {

template <typename T> struct KernelRNG;

namespace Access::KernelRNG {

template <typename T> struct Read {
  int stride;
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

} // namespace Access::KernelRNG

namespace ParticleLoopImplementation {

/**
 *  KernelParameter type for read-only access to a KernelRNG.
 */
template <typename T> struct KernelParameter<Access::Read<KernelRNG<T>>> {
  using type = Access::KernelRNG::Read<T>;
};

template <typename T> struct LoopParameter<Access::Read<KernelRNG<T>>> {
  using type = Access::KernelRNG::Read<T>;
};

template <typename T>
inline Access::KernelRNG::Read<T>
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                sycl::handler &cgh, Access::Read<KernelRNG<T> *> &a) {

  const auto host_arg = a.obj->impl_get_const(global_info);
  Access::KernelRNG::Read<T> kernel_arg = {std::get<0>(host_arg),
                                           std::get<1>(host_arg)};
  return kernel_arg;
}

template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              Access::KernelRNG::Read<T> &rhs,
                              Access::KernelRNG::Read<T> &lhs) {
  lhs = rhs;
}
template <typename T>
inline void pre_loop(ParticleLoopGlobalInfo *global_info,
                     Access::Read<KernelRNG<T> *> &arg) {
  arg.obj->impl_pre_loop_read(global_info);
}
template <typename T>
inline void post_loop(ParticleLoopGlobalInfo *global_info,
                      Access::Read<KernelRNG<T> *> &arg) {
  arg.obj->impl_post_loop_read(global_info);
}

} // namespace ParticleLoopImplementation

/**
 * Abstract base class for RNG implementations which create a block of random
 * numbers on the device prior to the loop execution.
 */
template <typename T> struct KernelRNG {

  /**
   * Create the loop arguments for the RNG implementation.
   *
   * @param global_info Global information for the loop about to be executed.
   * @returns Pointer to the device data that contains the RNG data.
   */
  virtual inline std::tuple<int, T *> impl_get_const(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) = 0;

  /**
   * Executed by the loop pre execution.
   * @param global_info Global information for the loop which is to be executed.
   */
  virtual inline void impl_pre_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) = 0;

  /**
   * Executed by the loop post execution.
   * @param global_info Global information for the loop which completed.
   */
  virtual inline void impl_post_loop_read(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info) = 0;
};

/**
 * TODO PLACE IN OWN FILE
 */
template <typename T> class HostKernelRNG : public KernelRNG<T> {
protected:
  int internal_state;
  std::map<SYCLTargetSharedPtr, std::shared_ptr<BufferDevice<T>>> d_buffers;

  inline T *allocate(SYCLTargetSharedPtr sycl_target, const int nrow) {
    if (nrow <= 0) {
      return nullptr;
    }
    const std::size_t required_size = nrow * this->num_components;

    if (!this->d_buffers.count(sycl_target)) {
      this->d_buffers[sycl_target] =
          std::make_unique<BufferDevice<T>>(sycl_target, required_size);
    } else {
      this->d_buffers.at(sycl_target)->realloc_no_copy(required_size, 1.2);
    }

    return this->d_buffers.at(sycl_target)->ptr;
  }

public:
  int num_components;
  int block_size;
  std::function<T()> generation_function;

  template <typename FUNC_TYPE>
  HostKernelRNG(FUNC_TYPE func, const int num_components,
                const int block_size = 4096)
      : generation_function(func), num_components(num_components),
        internal_state(0), block_size(block_size) {
    NESOASSERT(num_components > 0, "Cannot have a RNG for " +
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
    const int cell_start = global_info->starting_cell;
    const int cell_end = global_info->bounding_cell;
    int num_particles;
    if ((cell_end - cell_start) == 1) {
      // Single cell looping case
      // Allocate for all the particles in the cell. Slightly inefficient if
      // the loop is a particle sub group that only selects a small amount of
      // the cell.
      num_particles = global_info->particle_group->get_npart_cell(cell_start);
    } else {
      // Whole ParticleGroup looping case
      num_particles = global_info->particle_group->get_npart_local();
    }

    auto sycl_target = global_info->particle_group->sycl_target;

    return {num_particles, this->d_buffers.at(sycl_target)->ptr};
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

    const int cell_start = global_info->starting_cell;
    const int cell_end = global_info->bounding_cell;

    int num_particles;
    if ((cell_end - cell_start) == 1) {
      // Single cell looping case
      // Allocate for all the particles in the cell. Slightly inefficient if
      // the loop is a particle sub group that only selects a small amount of
      // the cell.
      num_particles = global_info->particle_group->get_npart_cell(cell_start);
    } else {
      // Whole ParticleGroup looping case
      num_particles = global_info->particle_group->get_npart_local();
    }

    // Allocate space
    auto sycl_target = global_info->particle_group->sycl_target;
    auto d_ptr = this->allocate(sycl_target, num_particles);
    auto d_ptr_start = d_ptr;

    // Create num_particles * num_components random numbers from the RNG
    const int num_random_numbers = num_particles * num_components;

    // Create the random number in blocks and copy to device blockwise.
    std::vector<T> block0(this->block_size);
    std::vector<T> block1(this->block_size);

    T *ptr_tmp;
    T *ptr_current = block0.data();
    T *ptr_next = block1.data();
    int num_numbers_moved = 0;

    sycl::event e;
    while (num_numbers_moved < num_random_numbers) {

      // Create a block of samples
      for (int ix = 0; ix < this->block_size; ix++) {
        ptr_current[ix] = this->generation_function();
      }

      // Wait until the previous block finished copying before starting this
      // copy
      e.wait_and_throw();
      const std::size_t num_to_memcpy =
          std::min(this->block_size, num_random_numbers - num_numbers_moved);
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

    NESOASSERT(num_numbers_moved == num_random_numbers,
               "Failed to copy the correct number of random numbers");
    NESOASSERT(d_ptr == d_ptr_start + num_random_numbers,
               "Failed to copy the correct number of random numbers (pointer "
               "arithmetic)");
  }

  /**
   * Ran executed by the loop post execution.
   * @param global_info Global information for the loop which completed.
   */
  virtual inline void impl_post_loop_read(
      [[maybe_unused]] ParticleLoopImplementation::ParticleLoopGlobalInfo
          *global_info) override {
    NESOASSERT(this->internal_state == 2,
               "HostKernelRNG Unexpected state, post loop called but internal "
               "state does not expect a loop to be running.");
    this->internal_state = 0;
  }
};

/**
 * TODO place in own file
 */
template <typename T, typename FUNC_TYPE>
inline std::shared_ptr<KernelRNG<T>> host_kernel_rng(FUNC_TYPE func,
                                                     const int num_components) {
  return std::dynamic_pointer_cast<KernelRNG<T>>(
      std::make_shared<HostKernelRNG<T>>(func, num_components));
}

} // namespace NESO::Particles

#endif
