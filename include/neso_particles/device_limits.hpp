#ifndef _NESO_PARTICLES_COMPUTE_TARGET_DEVICE_LIMITS_HPP_
#define _NESO_PARTICLES_COMPUTE_TARGET_DEVICE_LIMITS_HPP_

#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class DeviceLimits {
protected:
  sycl::range<1> max_global_workgroup_1;
  sycl::range<2> max_global_workgroup_2;
  sycl::range<3> max_global_workgroup_3;

  template <int N> inline sycl::range<N> get_max_global_workgroup();

  template <> inline sycl::range<1> get_max_global_workgroup() {
    return this->max_global_workgroup_1;
  }
  template <> inline sycl::range<2> get_max_global_workgroup() {
    return this->max_global_workgroup_2;
  }
  template <> inline sycl::range<3> get_max_global_workgroup() {
    return this->max_global_workgroup_3;
  }

  inline void setup_env() {
    constexpr std::size_t max_size_t = std::numeric_limits<std::size_t>::max();
    auto current_limits = this->get_max_global_workgroup<3>();
    const std::size_t env_max_0 = get_env_size_t(
        "NESO_PARTICLES_DEVICE_LIMIT_GLOBAL_SIZE_S0", current_limits.get(2));
    const std::size_t env_max_1 = get_env_size_t(
        "NESO_PARTICLES_DEVICE_LIMIT_GLOBAL_SIZE_S1", current_limits.get(1));
    const std::size_t env_max_2 = get_env_size_t(
        "NESO_PARTICLES_DEVICE_LIMIT_GLOBAL_SIZE_S2", current_limits.get(0));

    this->max_global_workgroup_1 = sycl::range<1>(env_max_0);
    this->max_global_workgroup_2 = sycl::range<2>(env_max_1, env_max_0);
    this->max_global_workgroup_3 =
        sycl::range<3>(env_max_2, env_max_1, env_max_0);
  }

  inline void setup_generic() {
    constexpr std::size_t max_size_t = std::numeric_limits<std::size_t>::max();
    this->max_global_workgroup_1 = sycl::range<1>(max_size_t);
    this->max_global_workgroup_2 = sycl::range<2>(max_size_t, max_size_t);
    this->max_global_workgroup_3 =
        sycl::range<3>(max_size_t, max_size_t, max_size_t);
  }

  inline void setup_nvidia() {
    this->max_global_workgroup_1 = sycl::range<1>(2147483647);
    this->max_global_workgroup_2 = sycl::range<2>(65535, 2147483647);
    this->max_global_workgroup_3 = sycl::range<3>(65535, 65535, 2147483647);
  }

public:
  sycl::device device;

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
    auto d1 = this->get_max_global_workgroup<1>();
    nprint("1D:", d1.get(0));
    auto d2 = this->get_max_global_workgroup<2>();
    nprint("2D:", d2.get(0), d2.get(1));
    auto d3 = this->get_max_global_workgroup<3>();
    nprint("3D:", d3.get(0), d3.get(1), d3.get(2));
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
    sycl::range<N> max_global_workgroup = this->get_max_global_workgroup<N>();
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
