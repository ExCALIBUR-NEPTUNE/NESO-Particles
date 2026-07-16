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

std::size_t MaskArray::get_num_masks_true(const std::size_t mask_index) {
  auto r0 = this->sycl_target->profile_map.start_region("MaskArray",
                                                        "get_num_masks_true");
  INT result = 0;

  if (this->size > 0) {
    MaskArrayDevice mad = this->get_device();

    auto d_INT = get_resource<BufferDevice<INT>,
                              ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    if (d_INT->size < 1) {
      d_INT->realloc_no_copy(1);
    }

    INT *k_int = d_INT->ptr;
    const auto k_size = this->size;

    const INT zero = 0;
    auto e0 = sycl_target->queue.memcpy(k_int, &zero, sizeof(INT));

    const std::size_t local_size =
        this->sycl_target->parameters
            ->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
            ->value;

    auto e1 = this->sycl_target->queue.parallel_for(
        this->sycl_target->device_limits.validate_nd_range(sycl::nd_range<1>(
            sycl::range<1>(local_size * div_round_up(this->size, local_size)),
            sycl::range<1>(local_size))),
        e0, [=](sycl::nd_item<1> idx) {
          int contrib = 0;
          const std::size_t gid = idx.get_global_linear_id();
          if (gid < k_size) {
            const bool v = mad.get(gid, mask_index);
            contrib = v ? 1 : 0;
          }

          const int group_reduce_v =
              reduce_over_group(idx.get_group(), contrib, sycl::plus{});

          if (idx.get_group().leader()) {
            atomic_fetch_add(k_int, static_cast<INT>(group_reduce_v));
          }
        });

    sycl_target->queue.memcpy(&result, k_int, sizeof(INT), e1).wait_and_throw();

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_INT);
  }
  this->sycl_target->profile_map.end_region(r0);
  return static_cast<std::size_t>(result);
}

namespace ParticleLoopImplementation {
Access::MaskArray::Read
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<MaskArray *> &a) {
  auto tmp = a.obj->get_device();
  return {tmp.d_masks, tmp.num_masks_per_entry, tmp.size};
}

Access::MaskArray::Write
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<MaskArray *> &a) {
  auto tmp = a.obj->get_device();
  return {tmp.d_masks, tmp.num_masks_per_entry, tmp.size};
}
} // namespace ParticleLoopImplementation

} // namespace NESO::Particles
