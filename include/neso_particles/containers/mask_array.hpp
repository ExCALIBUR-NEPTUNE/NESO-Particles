#ifndef _NESO_PARTICLES_MASK_ARRAY_HPP_
#define _NESO_PARTICLES_MASK_ARRAY_HPP_

#include <cstdint>
#include <memory>
#include <optional>

#include "../compute_target.hpp"
#include "../device_buffers.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../pair_loop/particle_pair_loop_base.hpp"

namespace NESO::Particles {

class MaskArray;

// The base integer type in which the masks are stored.
using MaskArrayBaseType = std::uint8_t;

/**
 * Device type for MaskArray.
 */
template <typename T> struct MaskArrayDeviceBase {
  // Number of bits per base element on the device.
  static constexpr MaskArrayBaseType num_bits_per_base =
      sizeof(MaskArrayBaseType) * CHAR_BIT;

  T d_masks{nullptr};
  std::size_t num_masks_per_entry{0};
  std::size_t size{0};

  /**
   * @param index_entry Index of the entry in [0, size)·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @returns The entry within a single MaskArrayBaseType that corresponds to
   * the provided entry and bit.
   */
  static inline MaskArrayBaseType
  get_inner_index([[maybe_unused]] const std::size_t index_entry,
                  const std::size_t index_bit) {
    return index_bit % num_bits_per_base;
  }

  /**
   * @param index_entry Index of the entry in [0, size)·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @returns The index to a MaskArrayBaseType.
   */
  static inline std::size_t
  get_outer_index([[maybe_unused]] const std::size_t index_entry,
                  const std::size_t index_bit) {
    return index_bit / num_bits_per_base;
  }

  /**
   * Get mask at location in base type.
   *
   * @param base Pointer to a base type instance·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @returns Value of mask (bit) at location.
   */
  static inline bool get_inner(MaskArrayBaseType const *const base,
                               const std::size_t index_bit) {

    const MaskArrayBaseType one_at_index = static_cast<MaskArrayBaseType>(1)
                                           << index_bit;
    const MaskArrayBaseType initial_base = *base;
    const MaskArrayBaseType masked_entry = initial_base & one_at_index;
    return masked_entry != 0;
  }

  /**
   * @param index_entry Index of the entry in [0, size)·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @returns Offset to the MaskArrayBaseType containing the mask.
   */
  inline std::size_t get_base_index(const std::size_t index_entry,
                                    const std::size_t index_bit) const {

    const std::size_t offset_bit =
        get_outer_index(index_entry, index_bit) * this->size;
    return offset_bit + index_entry;
  }

  /**
   * Get mask at location.
   *
   * @param index_entry Index of the entry in [0, size)·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @returns Value of mask (bit) at location.
   */
  inline bool get(const std::size_t index_entry,
                  const std::size_t index_bit) const {
    auto d_base = this->d_masks + this->get_base_index(index_entry, index_bit);
    return get_inner(d_base, get_inner_index(index_entry, index_bit));
  }
};

namespace Access::MaskArray {
using Read = MaskArrayDeviceBase<MaskArrayBaseType const * RESTRICT>;

struct Write : public MaskArrayDeviceBase<MaskArrayBaseType * RESTRICT> {

  /**
   * Set mask at location in base type.
   *
   * @param base Pointer to a base type instance·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @param value Value of mask (bit) to set.
   */
  static inline void set_inner(MaskArrayBaseType *base,
                               const std::size_t index_bit, const bool value) {

    const MaskArrayBaseType one_at_index = static_cast<MaskArrayBaseType>(1)
                                           << index_bit;

    const MaskArrayBaseType initial_base = *base;
    const MaskArrayBaseType set_true = one_at_index | initial_base;
    const MaskArrayBaseType set_false = (~one_at_index) & initial_base;
    *base = value ? set_true : set_false;
  }

  /**
   * Set mask at location.
   *
   * @param index_entry Index of the entry in [0, size)·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @param value Value of mask (bit) to set.
   */
  inline void set(const std::size_t index_entry, const std::size_t index_bit,
                  const bool value) const {
    auto d_base = this->d_masks + this->get_base_index(index_entry, index_bit);
    set_inner(d_base, get_inner_index(index_entry, index_bit), value);
  }
};

} // namespace Access::MaskArray

using MaskArrayDevice = Access::MaskArray::Write;

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a MaskArray.
 */
template <> struct LoopParameter<Access::Read<MaskArray>> {
  using type = Access::MaskArray::Read;
};

/**
 *  Loop parameter for write access of a MaskArray.
 */
template <> struct LoopParameter<Access::Write<MaskArray>> {
  using type = Access::MaskArray::Write;
};

/**
 *  KernelParameter type for read access to a MaskArray.
 */
template <> struct KernelParameter<Access::Read<MaskArray>> {
  using type = Access::MaskArray::Read;
};

/**
 *  KernelParameter type for write access to a MaskArray.
 */
template <> struct KernelParameter<Access::Write<MaskArray>> {
  using type = Access::MaskArray::Write;
};

/**
 *  Function to create the kernel argument for MaskArray read access.
 */
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::MaskArray::Read &rhs, Access::MaskArray::Read &lhs) {
  lhs = rhs;
}

/**
 *  Function to create the kernel argument for MaskArray write access.
 */
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  Access::MaskArray::Write &rhs,
                  Access::MaskArray::Write &lhs) {
  lhs = rhs;
}

} // namespace ParticleLoopImplementation

namespace ParticlePairLoopImplementation {

/**
 *  Function to create the kernel argument for MaskArray read
 * access in a pair loop.
 */
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    Access::MaskArray::Read &rhs, Access::MaskArray::Read &lhs) {
  lhs = rhs;
}

/**
 * Function to create the kernel argument for MaskArray write
 * access in a pair loop.
 */
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopIteration &iteration,
    [[maybe_unused]] ParticleLoopImplementation::ParticleLoopIteration
        &iteration_particle,
    Access::MaskArray::Write &rhs, Access::MaskArray::Write &lhs) {
  lhs = rhs;
}

} // namespace ParticlePairLoopImplementation

/**
 * Type that stores N bits per entry (particle).
 */
class MaskArray {
protected:
  std::shared_ptr<BufferDevice<MaskArrayBaseType>> d_masks;
  sycl::event event;

public:
  /**
   * @param value Mask value to set all masks to.
   * @returns A base element that is either all 1 or 0.
   */
  static MaskArrayBaseType get_reset_mask(const bool value);

  // Number of bits per base element on the device.
  static constexpr MaskArrayBaseType num_bits_per_base =
      sizeof(MaskArrayBaseType) * CHAR_BIT;

  MaskArray() = default;
  ~MaskArray();

  // Compute device.
  SYCLTargetSharedPtr sycl_target;

  // Number of masks per entry.
  std::size_t num_masks_per_entry{0};

  // Number of base elements per entry.
  std::size_t num_base_elements_per_entry{0};

  // Number of entries.
  std::size_t size{0};

  /**
   * Create a mask array on a given compute device with a set number of bits per
   * entry.
   *
   * @param sycl_target Compute device.
   * @param num_masks_per_entry Number of bits stored per entry.
   */
  MaskArray(SYCLTargetSharedPtr sycl_target,
            const std::size_t num_masks_per_entry);

  /**
   * Zero the masks optionally provide a new array size.
   *
   * @param reset_value Provide the mask value to reset all entries to.
   * @param new_size Optional new array size.
   */
  void reset(const bool value,
             std::optional<std::size_t> new_size = std::nullopt);

  /**
   * @returns Access to the masks via the device type.
   */
  MaskArrayDevice get_device();

  /**
   * @param mask_index Index of mask in entry to count, default 0.
   * @returns Number of masks set to true.
   */
  std::size_t get_num_masks_true(const std::size_t mask_index = 0);
};

namespace ParticleLoopImplementation {

/**
 * Method to compute access to a MaskArray (read)
 */
inline Access::MaskArray::Read
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<MaskArray *> &a) {
  auto tmp = a.obj->get_device();
  return {tmp.d_masks, tmp.num_masks_per_entry, tmp.size};
}

/**
 * Method to compute access to a MaskArray (read)
 */
inline Access::MaskArray::Write
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<MaskArray *> &a) {
  auto tmp = a.obj->get_device();
  return {tmp.d_masks, tmp.num_masks_per_entry, tmp.size};
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
