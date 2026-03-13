#include <neso_particles/compute_target.hpp>

namespace NESO::Particles {

int get_local_mpi_rank(MPI_Comm comm, int default_rank) {

  if (const char *env_char = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
    std::string env_str = std::string(env_char);
    const int env_int = std::stoi(env_str);
    return env_int;
  } else if (const char *env_char = std::getenv("MV2_COMM_WORLD_LOCAL_RANK")) {
    std::string env_str = std::string(env_char);
    const int env_int = std::stoi(env_str);
    return env_int;
  } else if (const char *env_char = std::getenv("MPI_LOCALRANKID")) {
    std::string env_str = std::string(env_char);
    const int env_int = std::stoi(env_str);
    return env_int;
  } else if (default_rank < 0) {
    CommPair comm_pair(comm);
    default_rank = comm_pair.rank_intra;
    comm_pair.free();
  }

  return default_rank;
}

std::size_t SYCLTarget::get_local_size() {
  const bool is_cpu = this->device.is_cpu();

  if (!is_cpu) {
    // If the device is a GPU then return 256 or the env size.
    const std::size_t env_local_size =
        get_env_size_t("NESO_PARTICLES_LOOP_LOCAL_SIZE", 256);
    return env_local_size;
  } else {
    // If the device is a CPU then return 32 or the env size.
    const std::size_t env_local_size =
        get_env_size_t("NESO_PARTICLES_LOOP_LOCAL_SIZE", 32);
    return env_local_size;
  }
}

void SYCLTarget::print_info_inner() {
  std::string mods = "";

#ifdef NESO_PARTICLES_SINGLE_COMPILED_LOOP
  mods += "single_compiled_loop ";
#endif

#ifdef NESO_PARTICLES_CAS_MAX_INT
  mods += "cas_max_INT ";
#endif
#ifdef NESO_PARTICLES_CAS_MIN_INT
  mods += "cas_min_INT ";
#endif
#ifdef NESO_PARTICLES_CAS_MAX_REAL
  mods += "cas_max_REAL ";
#endif
#ifdef NESO_PARTICLES_CAS_MIN_REAL
  mods += "cas_min_REAL ";
#endif
#ifdef NESO_PARTICLES_MPI_NEIGHBOUR_ALL_TO_ALL_FIX
  mods += "mpi_neighbour_all_to_all_fix ";
#endif

  if (device_aware_mpi_enabled()) {
    mods += "device_aware_mpi ";
  }

  std::size_t local_size =
      this->parameters->get<SizeTParameter>("LOOP_LOCAL_SIZE")->value;

  std::cout << "Using " << this->device.get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "Max compute units: "
            << this->device_limits.get_max_compute_units() << std::endl;
  std::cout << "Version: " << NESO_PARTICLES_VERSION_MAJOR << "."
            << NESO_PARTICLES_VERSION_MINOR << "."
            << NESO_PARTICLES_VERSION_PATCH << std::endl;
  std::cout << "In order queue: " << this->queue.is_in_order() << std::endl;
  std::cout << "Mods: " << mods << std::endl;
  std::cout << "MPI comm size      : " << this->comm_pair.size_parent
            << std::endl;
  std::cout << "MPI comm rank      : " << this->comm_pair.rank_parent
            << std::endl;
  std::cout << "MPI inter-comm size: " << this->comm_pair.size_inter
            << std::endl;
  std::cout << "MPI inter-comm rank: " << this->comm_pair.rank_inter
            << std::endl;
  std::cout << "MPI intra-comm size: " << this->comm_pair.size_intra
            << std::endl;
  std::cout << "MPI intra-comm rank: " << this->comm_pair.rank_intra
            << std::endl;
  std::cout << "MPI local rank     : " << this->local_rank << std::endl;
  std::cout << "SYCL device count: " << this->num_devices << std::endl;
  std::cout << "SYCL device index: " << this->device_index << std::endl;
  std::cout << "SYCL nd_range local_size  : " << local_size << std::endl;
  std::cout << "SYCL device cacheline size: "
            << this->device_limits.get_cacheline_size() << std::endl;
  this->device_limits.print();
}

SYCLTarget::SYCLTarget(const int gpu_device, MPI_Comm comm, int local_rank)
    : local_rank(local_rank), comm_pair(comm),
      resource_stack_map(std::make_shared<ResourceStackMap>()) {
  if (gpu_device > 0) {
    try {
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
      this->device = sycl::device{sycl::gpu_selector()};
#else
      this->device = sycl::device{sycl::gpu_selector_v};
#endif
    } catch (sycl::exception const &e) {
      std::cout << "Cannot select a GPU\n" << e.what() << "\n";
      std::cout << "Using a CPU device\n";
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
      this->device = sycl::device{sycl::cpu_selector()};
#else
      this->device = sycl::device{sycl::cpu_selector_v};
#endif
    }
  } else if (gpu_device < 0) {
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
    this->device = sycl::device{sycl::cpu_selector()};
#else
    this->device = sycl::device{sycl::cpu_selector_v};
#endif
  } else {

    // Get the default device and platform as they are most likely to be the
    // desired device based on SYCL implementation/runtime/environment
    // variables.
#ifdef NESO_PARTICLES_LEGACY_DEVICE_SELECTORS
    sycl::device default_device{sycl::default_selector()};
#else
    sycl::device default_device{sycl::default_selector_v};
#endif
    auto default_platform = default_device.get_platform();

    // Get all devices from the default platform
    auto devices = default_platform.get_devices();

    // determine the local rank to use for round robin device assignment.
    if (this->local_rank < 0) {
      this->local_rank = get_local_mpi_rank(comm, this->comm_pair.rank_intra);
    }

    // round robin assign devices to local MPI ranks.
    this->num_devices = devices.size();
    this->device_index = this->local_rank % this->num_devices;
    this->device = devices[this->device_index];
    this->device_limits = DeviceLimits(this->device);

    this->profile_map.set("MPI", "MPI_COMM_WORLD_rank_local", this->local_rank);
    this->profile_map.set("SYCL", "DEVICE_COUNT", this->num_devices);
    this->profile_map.set("SYCL", "DEVICE_INDEX", this->device_index);
    this->profile_map.set("SYCL",
                          this->device.get_info<sycl::info::device::name>(), 0);

    // Setup the parameter store
    this->parameters = std::make_shared<Parameters>();
    this->parameters->set("LOOP_LOCAL_SIZE", std::make_shared<SizeTParameter>(
                                                 this->get_local_size()));
    this->parameters->set("LOOP_NBIN",
                          std::make_shared<SizeTParameter>(
                              get_env_size_t("NESO_PARTICLES_LOOP_NBIN", 4)));
    this->parameters->set("MAX_COMPUTE_UNITS",
                          std::make_shared<SizeTParameter>(
                              this->device_limits.get_max_compute_units()));
  }

  if (get_env_size_t("NESO_PARTICLES_IN_ORDER_QUEUE", 0)) {
    this->queue =
        sycl::queue(this->device, {sycl::property::queue::in_order{}});
  } else {
    this->queue = sycl::queue(this->device);
  }
  this->comm = comm;

  this->profile_map.set("MPI", "MPI_COMM_WORLD_rank",
                        this->comm_pair.rank_parent);
  this->profile_map.set("MPI", "MPI_COMM_WORLD_size",
                        this->comm_pair.size_parent);

  if (get_env_size_t("NESO_PARTICLES_VERBOSE_DEVICE", 0)) {
    this->print_world_device_info();
  }

  this->auto_profiling_prefix =
      get_env_string("NESO_PARTICLES_AUTO_PROFILE", "");

  if (this->auto_profiling_prefix.size()) {
    this->profile_map.enable();
  }

#ifdef DEBUG_OOB_CHECK
  for (int cx = 0; cx < DEBUG_OOB_WIDTH; cx++) {
    this->ptr_bit_mask[cx] = static_cast<unsigned char>(255);
  }
#endif

// The tests run these checks as a test, hence don't redo them all the time.
#ifdef NESO_PARTICLES_TEST_COMPILATION
#define NESO_PARTICLES_DISABLE_ATOMIC_SELFTEST
#endif

#ifndef NESO_PARTICLES_DISABLE_ATOMIC_SELFTEST
  this->device_limits.check_atomics_sanity(this->queue);
#endif
}

void SYCLTarget::print_device_info() {
  if (this->comm_pair.rank_parent == 0) {
    this->print_info_inner();
  }
}

void SYCLTarget::print_world_device_info() {
  int size = this->comm_pair.size_parent;
  int rank = this->comm_pair.rank_parent;
  for (int rx = 0; rx < size; rx++) {
    if (rx == rank) {
      std::cout << "---------------------------------------------------------"
                   "-----------------------"
                << std::endl;
      this->print_info_inner();
    }
    std::cout << std::flush;
    MPI_Barrier(this->comm);
  }
  if (!rank) {
    std::cout << "-----------------------------------------------------------"
                 "---------------------"
              << std::endl
              << std::flush;
  }
  std::cout << std::flush;
  MPI_Barrier(this->comm);
}

void SYCLTarget::free() {
  if (this->auto_profiling_prefix.size()) {
    this->profile_map.write_events_json(this->auto_profiling_prefix,
                                        this->comm_pair.rank_parent);
  }

  this->resource_stack_map->free();
  this->comm_pair.free();
}

void *SYCLTarget::malloc_device(const std::size_t size_bytes,
                                const std::size_t align_bytes) {

  auto lambda_alloc = [&](const std::size_t b) -> void * {
    if (align_bytes) {
      return sycl::aligned_alloc_device(
          align_bytes, get_next_multiple(b, align_bytes), this->queue);
    } else {
      return sycl::malloc_device(b, this->queue);
    }
  };

#ifndef DEBUG_OOB_CHECK
  void *ptr = lambda_alloc(size_bytes);
  return ptr;
#else
  unsigned char *ptr =
      (unsigned char *)lambda_alloc(size_bytes + 2 * DEBUG_OOB_WIDTH);

  unsigned char *ptr_user = ptr + DEBUG_OOB_WIDTH;
  this->ptr_map[ptr_user] = size_bytes;
  NESOASSERT(ptr != nullptr, "pad pointer from malloc_device");

  this->queue.memcpy(ptr, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH).wait();

  this->queue
      .memcpy(ptr_user + size_bytes, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH)
      .wait();

  return (void *)ptr_user;
#endif
}

void *SYCLTarget::malloc_shared(const std::size_t size_bytes,
                                const std::size_t align_bytes) {

  auto lambda_alloc = [&](const std::size_t b) -> void * {
    if (align_bytes) {
      return sycl::aligned_alloc_shared(
          align_bytes, get_next_multiple(b, align_bytes), this->queue);
    } else {
      return sycl::malloc_shared(b, this->queue);
    }
  };

#ifndef DEBUG_OOB_CHECK
  return lambda_alloc(size_bytes);
#else
  unsigned char *ptr =
      (unsigned char *)lambda_alloc(size_bytes + 2 * DEBUG_OOB_WIDTH);

  unsigned char *ptr_user = ptr + DEBUG_OOB_WIDTH;
  this->ptr_map[ptr_user] = size_bytes;
  NESOASSERT(ptr != nullptr, "pad pointer from malloc_shared");

  this->queue.memcpy(ptr, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH).wait();

  this->queue
      .memcpy(ptr_user + size_bytes, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH)
      .wait();

  return (void *)ptr_user;
#endif
}

void *SYCLTarget::malloc_host(const std::size_t size_bytes,
                              const std::size_t align_bytes) {

  auto lambda_alloc = [&](const std::size_t b) -> void * {
    if (align_bytes) {
      auto ptr = sycl::aligned_alloc_host(
          align_bytes, get_next_multiple(b, align_bytes), this->queue);
      return ptr;
    } else {
      return sycl::malloc_host(b, this->queue);
    }
  };
#ifndef DEBUG_OOB_CHECK
  return lambda_alloc(size_bytes);
#else

  unsigned char *ptr =
      (unsigned char *)lambda_alloc(size_bytes + 2 * DEBUG_OOB_WIDTH);
  unsigned char *ptr_user = ptr + DEBUG_OOB_WIDTH;
  this->ptr_map[ptr_user] = size_bytes;
  NESOASSERT(ptr != nullptr, "pad pointer from malloc_host");

  this->queue.memcpy(ptr, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH).wait();

  this->queue
      .memcpy(ptr_user + size_bytes, this->ptr_bit_mask.data(), DEBUG_OOB_WIDTH)
      .wait();

  return (void *)ptr_user;
  // return ptr;
#endif
}

void SYCLTarget::check_ptrs() {
  for (auto &px : this->ptr_map) {
    this->check_ptr(px.first, px.second);
  }
}

void SYCLTarget::check_ptr([[maybe_unused]] unsigned char *ptr_user,
                           [[maybe_unused]] const std::size_t size_bytes) {

#ifdef DEBUG_OOB_CHECK
  this->queue
      .memcpy(this->ptr_bit_tmp.data(), ptr_user - DEBUG_OOB_WIDTH,
              DEBUG_OOB_WIDTH)
      .wait();

  for (int cx = 0; cx < DEBUG_OOB_WIDTH; cx++) {
    NESOASSERT(this->ptr_bit_tmp[cx] == static_cast<unsigned char>(255),
               "DEBUG PADDING START TOUCHED");
  }

  this->queue
      .memcpy(this->ptr_bit_tmp.data(), ptr_user + size_bytes, DEBUG_OOB_WIDTH)
      .wait();

  for (int cx = 0; cx < DEBUG_OOB_WIDTH; cx++) {
    NESOASSERT(this->ptr_bit_tmp[cx] == static_cast<unsigned char>(255),
               "DEBUG PADDING END TOUCHED");
  }

#endif
}

std::size_t
SYCLTarget::get_num_local_work_items(const std::size_t num_bytes_offset,
                                     const std::size_t num_bytes,
                                     const std::size_t default_num) {
  if (num_bytes <= 0) {
    return default_num;
  } else {
    const std::size_t local_mem_size = this->device_limits.local_mem_size;
    NESOASSERT(local_mem_size >= num_bytes_offset,
               "Offset bytes is larger than local memory.");
    const std::size_t max_num_workitems =
        (local_mem_size - num_bytes_offset) / num_bytes;
    // find the max power of two that does not exceed the number of work
    // items.
    const std::size_t two_power = std::log2(max_num_workitems);
    const std::size_t max_base_two_num_workitems = std::pow(2, two_power);

    const std::size_t deduced_num_work_items =
        std::min(default_num, max_base_two_num_workitems);
    NESOASSERT((deduced_num_work_items > 0),
               "Deduced number of work items is not strictly positive.");

    const std::size_t local_mem_bytes =
        deduced_num_work_items * num_bytes + num_bytes_offset;
    NESOASSERT(local_mem_size >= local_mem_bytes, "Not enough local memory");
    return deduced_num_work_items;
  }
}

std::size_t
SYCLTarget::get_num_local_work_items(const std::size_t num_bytes,
                                     const std::size_t default_num) {
  return get_num_local_work_items(0, num_bytes, default_num);
}

NDRangePeel1D get_nd_range_peel_1d(const std::size_t size,
                                   const std::size_t local_size) {
  const auto div_mod = std::div(static_cast<long long>(size),
                                static_cast<long long>(local_size));

  const std::size_t outer_size =
      static_cast<std::size_t>(div_mod.quot) * local_size;
  const bool peel_exists = !(div_mod.rem == 0);
  const std::size_t outer_size_peel = (peel_exists ? 1 : 0) * local_size;
  const std::size_t offset = outer_size;

  return NDRangePeel1D{
      sycl::nd_range<1>(sycl::range<1>(outer_size), sycl::range<1>(local_size)),
      peel_exists, offset,
      sycl::nd_range<1>(sycl::range<1>(outer_size_peel),
                        sycl::range<1>(local_size))};
}

template sycl::event joint_exclusive_scan(SYCLTargetSharedPtr sycl_target,
                                          std::size_t N, int *d_src,
                                          int *d_dst);
template sycl::event joint_exclusive_scan(SYCLTargetSharedPtr sycl_target,
                                          std::size_t N, INT *d_src,
                                          INT *d_dst);

std::size_t
get_joint_exclusive_scan_aux_num_blocks(SYCLTargetSharedPtr sycl_target,
                                        const std::size_t N) {

  const std::size_t local_size =
      sycl_target->parameters->template get<SizeTParameter>("LOOP_LOCAL_SIZE")
          ->value;

  const std::size_t max_compute_units =
      sycl_target->parameters->template get<SizeTParameter>("MAX_COMPUTE_UNITS")
          ->value;

  return std::min(div_round_up(N, local_size), max_compute_units);
}

std::size_t
get_joint_exclusive_scan_aux_array_size(SYCLTargetSharedPtr sycl_target,
                                        const std::size_t N) {
  return 2 * get_joint_exclusive_scan_aux_num_blocks(sycl_target, N);
}

template sycl::event
joint_exclusive_scan_n_sum(SYCLTargetSharedPtr sycl_target, std::size_t N,
                           const int *RESTRICT const d_array_sizes,
                           const int *RESTRICT const d_array_offsets,
                           int *d_src, int *d_dst, int *d_dst_sum);

template sycl::event
joint_exclusive_scan_n_sum(SYCLTargetSharedPtr sycl_target, std::size_t N,
                           const INT *RESTRICT const d_array_sizes,
                           const INT *RESTRICT const d_array_offsets,
                           INT *d_src, INT *d_dst, INT *d_dst_sum);

template sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       const int *RESTRICT const d_array_sizes,
                       const int *RESTRICT const d_array_offsets, int *d_src,
                       int *d_dst);

template sycl::event
joint_exclusive_scan_n(SYCLTargetSharedPtr sycl_target, std::size_t N,
                       const INT *RESTRICT const d_array_sizes,
                       const INT *RESTRICT const d_array_offsets, INT *d_src,
                       INT *d_dst);

template sycl::event matrix_transpose(SYCLTargetSharedPtr sycl_target,
                                      const std::size_t num_rows,
                                      const std::size_t num_cols,
                                      const REAL *RESTRICT const d_src,
                                      REAL *RESTRICT d_dst);
} // namespace NESO::Particles
