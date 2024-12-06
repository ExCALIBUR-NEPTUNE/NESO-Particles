#ifndef _NESO_PARTICLES_COMPUTE_TARGET_DEVICE_LIMITS_HPP_
#define _NESO_PARTICLES_COMPUTE_TARGET_DEVICE_LIMITS_HPP_

#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

namespace Private {

struct WorkGroupLimits {
  sycl::range<1> max_global_workgroup_1;
  sycl::range<2> max_global_workgroup_2;
  sycl::range<3> max_global_workgroup_3;
};

template <int N>
inline sycl::range<N> get_max_global_workgroup(WorkGroupLimits &wgl);

template <>
inline sycl::range<1> get_max_global_workgroup(WorkGroupLimits &wgl) {
  return wgl.max_global_workgroup_1;
}
template <>
inline sycl::range<2> get_max_global_workgroup(WorkGroupLimits &wgl) {
  return wgl.max_global_workgroup_2;
}
template <>
inline sycl::range<3> get_max_global_workgroup(WorkGroupLimits &wgl) {
  return wgl.max_global_workgroup_3;
}

} // namespace Private

class DeviceLimits {
protected:
  Private::WorkGroupLimits wgl;

  inline void setup_env() {
    auto current_limits = Private::get_max_global_workgroup<3>(wgl);
    const std::size_t env_max_0 = get_env_size_t(
        "NESO_PARTICLES_DEVICE_LIMIT_GLOBAL_SIZE_S0", current_limits.get(2));
    const std::size_t env_max_1 = get_env_size_t(
        "NESO_PARTICLES_DEVICE_LIMIT_GLOBAL_SIZE_S1", current_limits.get(1));
    const std::size_t env_max_2 = get_env_size_t(
        "NESO_PARTICLES_DEVICE_LIMIT_GLOBAL_SIZE_S2", current_limits.get(0));
    this->local_mem_size = get_env_size_t(
        "NESO_PARTICLES_DEVICE_LIMIT_LOCAL_MEM_SIZE", this->local_mem_size);

    this->wgl.max_global_workgroup_1 = sycl::range<1>(env_max_0);
    this->wgl.max_global_workgroup_2 = sycl::range<2>(env_max_1, env_max_0);
    this->wgl.max_global_workgroup_3 =
        sycl::range<3>(env_max_2, env_max_1, env_max_0);
  }

  inline void setup_generic() {
    constexpr std::size_t max_size_t = std::numeric_limits<std::size_t>::max();
    this->wgl.max_global_workgroup_1 = sycl::range<1>(max_size_t);
    this->wgl.max_global_workgroup_2 = sycl::range<2>(max_size_t, max_size_t);
    this->wgl.max_global_workgroup_3 =
        sycl::range<3>(max_size_t, max_size_t, max_size_t);
    auto local_mem_exists =
        device.get_info<sycl::info::device::local_mem_type>() !=
        sycl::info::local_mem_type::none;

    NESOASSERT(local_mem_exists, "Local memory does not exist.");
    this->local_mem_size =
        device.get_info<sycl::info::device::local_mem_size>();
  }

  inline void setup_nvidia() {
    this->wgl.max_global_workgroup_1 = sycl::range<1>(2147483647);
    this->wgl.max_global_workgroup_2 = sycl::range<2>(65535, 2147483647);
    this->wgl.max_global_workgroup_3 = sycl::range<3>(65535, 65535, 2147483647);
  }

public:
  sycl::device device;
  std::size_t local_mem_size;

  DeviceLimits() = default;
  DeviceLimits(sycl::device device) : device(device) {
    this->setup_generic();
    auto vendor_name = device.get_info<sycl::info::device::vendor>();
    if (vendor_name == "NVIDIA") {
      this->setup_nvidia();
    }
    this->setup_env();
  }

  inline void print() {
    nprint("Using global workgroup limits:");
    auto d1 = Private::get_max_global_workgroup<1>(wgl);
    nprint("1D:", d1.get(0));
    auto d2 = Private::get_max_global_workgroup<2>(wgl);
    nprint("2D:", d2.get(0), d2.get(1));
    auto d3 = Private::get_max_global_workgroup<3>(wgl);
    nprint("3D:", d3.get(0), d3.get(1), d3.get(2));
    auto local_mem_exists =
        this->device.get_info<sycl::info::device::local_mem_type>() !=
        sycl::info::local_mem_type::none;
    nprint("Local memory exists:", static_cast<int>(local_mem_exists));
    auto local_mem_size_device =
        device.get_info<sycl::info::device::local_mem_size>();
    nprint("Local memory size  :", local_mem_size_device);
    nprint("Local memory in use:", this->local_mem_size);
  }

  /**
   * Validate that a range is valid as a global work group.
   *
   * @param range_global Iteration set to validate.
   * @returns range_global if validated.
   */
  template <int N>
  inline sycl::range<N>
  validate_range_global(const sycl::range<N> &range_global) {
    sycl::range<N> max_global_workgroup =
        Private::get_max_global_workgroup<N>(wgl);
    for (int dx = 0; dx < N; dx++) {
      if (max_global_workgroup.get(dx)) {
        NESOASSERT(
            range_global.get(dx) <= max_global_workgroup.get(dx),
            "Workgroup size exceeds device maximum global workgroup size.");
      }
    }
    return range_global;
  }

  /**
   * Validate that a range is valid as a local work group.
   *
   * @param range_local Iteration set to validate.
   * @returns range_local if validated.
   */
  template <int N>
  inline sycl::range<N>
  validate_range_local(const sycl::range<N> &range_local) {

    const sycl::range<N> max_work_item_sizes =
        this->device.get_info<sycl::info::device::max_work_item_sizes<N>>();
    for (int dx = 0; dx < N; dx++) {
      if (max_work_item_sizes.get(dx)) {
        NESOASSERT(
            range_local.get(dx) <= max_work_item_sizes.get(dx),
            "Workgroup size exceeds device maximum local workgroup size.");
      }
    }
    return range_local;
  }

  /**
   * Validate that an nd_range is valid. Error if not.
   *
   * @param is Iteration set to validate.
   * @returns nd_range if validated.
   */
  template <int N>
  inline sycl::nd_range<N> validate_nd_range(const sycl::nd_range<N> &is) {

    // Test the global range is valid.
    const auto range_global = is.get_global_range();
    this->validate_range_global(range_global);

    // Test the local range is valid.
    const auto range_local = is.get_local_range();
    this->validate_range_local(range_local);

    // Test local range is a factor of global range
    for (int dx = 0; dx < N; dx++) {
      auto pq = std::div(static_cast<long long>(range_global.get(dx)),
                         static_cast<long long>(range_local.get(dx)));
      NESOASSERT(pq.rem == 0,
                 "Local workgroup does not factor global workgroup.");
    }
    return is;
  }
};

} // namespace NESO::Particles

#endif
