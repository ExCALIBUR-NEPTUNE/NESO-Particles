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

// The base integer type in which the masks are stored.
using MaskArrayBaseType = std::uint32_t;

/**
 * Device type for MaskArray.
 */
struct MaskArrayDevice {
  // Number of bits per base element on the device.
  static constexpr MaskArrayBaseType num_bits_per_base =
      sizeof(MaskArrayBaseType) * CHAR_BIT;

  MaskArrayBaseType const *d_masks{nullptr};
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
  get_inner_index(const std::size_t index_entry,
                  [[maybe_unused]] const std::size_t index_bit) {
    return index_entry % num_bits_per_base;
  }

  /**
   * @param index_entry Index of the entry in [0, size)·
   * @param index_bit Index of the mask (bit) in the entry in [0,
   * num_masks_per_entry).
   * @returns The index to a MaskArrayBaseType.
   */
  static inline MaskArrayBaseType
  get_outer_index(const std::size_t index_entry,
                  [[maybe_unused]] const std::size_t index_bit) {
    return index_entry / num_bits_per_base;
  }
};

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

  /**
   * @param N Number of elements.
   * @param num_masks_per_entry Number of bits per element.
   * @returns The number of MaskArrayBaseTypes required for a given number of
   * elements and bits (masks) per element.
   */
  static std::size_t
  get_num_base_elements(const std::size_t N,
                        const std::size_t num_masks_per_entry);

  MaskArray() = default;
  ~MaskArray();

  // Compute device.
  SYCLTargetSharedPtr sycl_target;

  // Number of masks per entry.
  std::size_t num_masks_per_entry{0};

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
};

} // namespace NESO::Particles

#endif
