#ifndef _NESO_PARTICLES_COMPUTE_TARGET_DEVICE_LIMITS_HPP_
#define _NESO_PARTICLES_COMPUTE_TARGET_DEVICE_LIMITS_HPP_

#include "device_atomic_sanity_check.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"
#include <set>
#include <vector>

namespace NESO::Particles {

namespace Private {

struct WorkGroupLimits {
  sycl::range<1> max_global_workgroup_1;
  sycl::range<2> max_global_workgroup_2;
  sycl::range<3> max_global_workgroup_3;
  std::size_t max_work_group_size;
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

  void setup_env();
  void setup_generic();
  void setup_nvidia();

public:
  sycl::device device;
  std::size_t local_mem_size;
  std::set<std::vector<std::size_t>> validated_types;

  DeviceLimits() = default;
  DeviceLimits(sycl::device device) : device(device) {
    this->setup_generic();
    auto vendor_name = device.get_info<sycl::info::device::vendor>();
    if (vendor_name == "NVIDIA") {
      this->setup_nvidia();
    }
    this->setup_env();
  }

  void print();

  /**
   * @returns The native vector width for REAL.
   */
  inline std::size_t get_native_vector_width_real() {
    if constexpr (std::is_same_v<REAL, double>) {
      return static_cast<std::size_t>(
          device.get_info<sycl::info::device::native_vector_width_double>());
    } else {
      return static_cast<std::size_t>(
          device.get_info<sycl::info::device::native_vector_width_float>());
    }
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
        Private::get_max_global_workgroup<N>(this->wgl);
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
    std::size_t total_size = 1;
    for (int dx = 0; dx < N; dx++) {
      const std::size_t dx_size = range_local.get(dx);
      if (max_work_item_sizes.get(dx)) {
        NESOASSERT(dx_size <= max_work_item_sizes.get(dx),
                   "Workgroup size exceeds device maximum local workgroup size "
                   "in a dimension.");
      }
      total_size *= dx_size;
    }
    NESOASSERT(total_size <= this->wgl.max_work_group_size,
               "Workgroup size exceeds device maximum local workgroup size.");
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

  /**
   * Check atomics pass basic functionality tests.
   *
   * @param queue sycl::queue to run tests on.
   * @param fatal Call NESOASSERT if any of the tests fail, default true.
   * @returns True if tests pass.
   */
  bool check_atomics_sanity(sycl::queue queue, const bool fatal = true);

  /**
   * Get the cacheline size in either bytes or as a rounded up multiple of a
   * number of bytes.
   *
   * @param num_bytes Default 1, optionally specify a factor such that if the
   * cacheline is N bytes then this function returns M such that M * num_bytes
   * >= N.
   * @returns Cacheline size in bytes or multiple of provided factor.
   */
  std::size_t get_cacheline_size(const std::size_t num_bytes = 1);
};

} // namespace NESO::Particles

#endif
