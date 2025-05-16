#include <neso_particles/device_limits.hpp>

namespace NESO::Particles {

void DeviceLimits::setup_env() {
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
  this->wgl.max_work_group_size =
      get_env_size_t("NESO_PARTICLES_DEVICE_LIMIT_WORK_GROUP_SIZE",
                     this->wgl.max_work_group_size);
}

void DeviceLimits::setup_generic() {
  constexpr std::size_t max_size_t = std::numeric_limits<std::size_t>::max();
  this->wgl.max_global_workgroup_1 = sycl::range<1>(max_size_t);
  this->wgl.max_global_workgroup_2 = sycl::range<2>(max_size_t, max_size_t);
  this->wgl.max_global_workgroup_3 =
      sycl::range<3>(max_size_t, max_size_t, max_size_t);
  this->wgl.max_work_group_size =
      device.get_info<sycl::info::device::max_work_group_size>();

  auto local_mem_exists =
      device.get_info<sycl::info::device::local_mem_type>() !=
      sycl::info::local_mem_type::none;

  NESOASSERT(local_mem_exists, "Local memory does not exist.");
  this->local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
}

void DeviceLimits::setup_nvidia() {
  this->wgl.max_global_workgroup_1 = sycl::range<1>(2147483647);
  this->wgl.max_global_workgroup_2 = sycl::range<2>(65535, 2147483647);
  this->wgl.max_global_workgroup_3 = sycl::range<3>(65535, 65535, 2147483647);
  this->wgl.max_work_group_size = 1024;
}

void DeviceLimits::print() {
  nprint("Using global workgroup limits:");
  auto d1 = Private::get_max_global_workgroup<1>(this->wgl);
  nprint("1D:", d1.get(0));
  auto d2 = Private::get_max_global_workgroup<2>(this->wgl);
  nprint("2D:", d2.get(0), d2.get(1));
  auto d3 = Private::get_max_global_workgroup<3>(this->wgl);
  nprint("3D:", d3.get(0), d3.get(1), d3.get(2));
  nprint("Workgroup size limit:", this->wgl.max_work_group_size);
  auto local_mem_exists =
      this->device.get_info<sycl::info::device::local_mem_type>() !=
      sycl::info::local_mem_type::none;
  nprint("Local memory exists:", static_cast<int>(local_mem_exists));
  auto local_mem_size_device =
      device.get_info<sycl::info::device::local_mem_size>();
  nprint("Local memory size  :", local_mem_size_device);
  nprint("Local memory in use:", this->local_mem_size);
}

bool DeviceLimits::check_atomics_sanity(sycl::queue queue, const bool fatal) {
  bool success = true;

  success = success && atomic_binop_check(queue, CheckAdd<int>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckAdd<int> failed.");
  success = success && atomic_binop_check(queue, CheckAdd<INT>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckAdd<INT> failed.");
  success = success && atomic_binop_check(queue, CheckAdd<REAL>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckAdd<REAL> failed.");

  success = success && atomic_binop_check(queue, CheckMin<int>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckMin<int> failed.");
  success = success && atomic_binop_check(queue, CheckMin<INT>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckMin<INT> failed.");
  success = success && atomic_binop_check(queue, CheckMin<REAL>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckMin<REAL> failed.");

  success = success && atomic_binop_check(queue, CheckMax<int>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckMax<int> failed.");
  success = success && atomic_binop_check(queue, CheckMax<INT>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckMax<INT> failed.");
  success = success && atomic_binop_check(queue, CheckMax<REAL>{});
  NESOASSERT((!fatal) || success,
             "Atomic sanity check on CheckMax<REAL> failed.");

  return success;
}

std::size_t DeviceLimits::get_cacheline_size(const std::size_t num_bytes) {
  const std::size_t hardware_cacheline_size =
      this->device.get_info<sycl::info::device::global_mem_cache_line_size>();
  NESOASSERT(hardware_cacheline_size > 0,
             "Bad cacheline size reported by SYCL implementation.");
  return get_next_multiple(hardware_cacheline_size, num_bytes);
}

} // namespace NESO::Particles
