#include <neso_particles/containers/mask_array.hpp>

namespace NESO::Particles {

MaskArray::~MaskArray() { this->event.wait(); }

MaskArray::MaskArray(SYCLTargetSharedPtr sycl_target,
                     const std::size_t num_masks_per_entry)
    : sycl_target(sycl_target), num_masks_per_entry(num_masks_per_entry) {
  NESOASSERT(sycl_target != nullptr, "Bad compute device");

  this->num_base_elements_per_entry = div_round_up(
      num_masks_per_entry, static_cast<std::size_t>(num_bits_per_base));
}

MaskArrayBaseType MaskArray::get_reset_mask(const bool value) {
  constexpr MaskArrayBaseType reset_base{0};
  const MaskArrayBaseType reset_value = value ? ~reset_base : reset_base;
  return reset_value;
}

void MaskArray::reset(const bool value, std::optional<std::size_t> new_size) {
  this->event.wait_and_throw();

  if (new_size != std::nullopt) {
    this->size = new_size.value();
    const std::size_t total_length =
        this->num_base_elements_per_entry * this->size;

    if (total_length) {
      if (this->d_masks == nullptr) {
        this->d_masks = std::make_shared<BufferDevice<MaskArrayBaseType>>(
            this->sycl_target, total_length);
      } else {
        this->d_masks->realloc_no_copy(total_length);
      }
    }
  }

  if (this->d_masks != nullptr) {
    const std::size_t total_length =
        this->num_base_elements_per_entry * this->size;
    const MaskArrayBaseType reset_value = this->get_reset_mask(value);
    if (total_length) {
      this->event = this->sycl_target->queue.fill<MaskArrayBaseType>(
          this->d_masks->ptr, reset_value, total_length);
    }
  }
}

MaskArrayDevice MaskArray::get_device() {
  this->event.wait_and_throw();
  return {this->d_masks != nullptr ? this->d_masks->ptr : nullptr,
          this->num_masks_per_entry, this->size};
}

} // namespace NESO::Particles
