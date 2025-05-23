#include <neso_particles/common_impl.hpp>
#include <neso_particles/loop/particle_loop_iteration_set.hpp>
#include <neso_particles/particle_dat.hpp>

namespace NESO::Particles {
namespace ParticleLoopImplementation {

ParticleLoopIterationSet::ParticleLoopIterationSet(const int nbin,
                                                   const int ncell,
                                                   int *h_npart_cell)
    : nbin(std::min(ncell, nbin)), ncell(ncell), h_npart_cell(h_npart_cell) {
  this->iteration_set.reserve(nbin);
  this->cell_offsets.reserve(nbin);
}

std::tuple<int, std::vector<sycl::nd_range<2>> &, std::vector<std::size_t> &>
ParticleLoopIterationSet::get(const std::optional<int> cell,
                              const size_t local_size) {

  this->iteration_set.clear();
  this->cell_offsets.clear();
  this->iteration_set_size = 0;

  if (cell == std::nullopt) {
    for (int binx = 0; binx < nbin; binx++) {
      int start, end;
      get_decomp_1d(nbin, ncell, binx, &start, &end);
      const int bin_width = end - start;
      int cell_maxi = 0;
      int cell_avg = 0;
      for (int cellx = start; cellx < end; cellx++) {
        const int cell_occ = h_npart_cell[cellx];
        cell_maxi = std::max(cell_maxi, cell_occ);
        cell_avg += cell_occ;
        this->iteration_set_size += cell_occ;
      }
      cell_avg = (((REAL)cell_avg) / ((REAL)(end - start)));
      const size_t cell_local_size =
          get_min_power_of_two((size_t)cell_avg, local_size);
      const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                    static_cast<long long>(cell_local_size));
      const std::size_t outer_size =
          static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1)) *
          cell_local_size;

      if (cell_maxi > 0) {
        this->iteration_set.emplace_back(
            sycl::nd_range<2>(sycl::range<2>(bin_width, outer_size),
                              sycl::range<2>(1, cell_local_size)));
        this->cell_offsets.push_back(static_cast<std::size_t>(start));
      }
    }

    return {this->iteration_set.size(), this->iteration_set,
            this->cell_offsets};
  } else {
    const int cellx = cell.value();
    const size_t cell_maxi = static_cast<size_t>(h_npart_cell[cellx]);

    this->iteration_set_size = cell_maxi;
    const size_t cell_local_size =
        get_min_power_of_two((size_t)cell_maxi, local_size);

    const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                  static_cast<long long>(cell_local_size));
    const std::size_t outer_size =
        static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1)) *
        cell_local_size;
    this->iteration_set.emplace_back(sycl::nd_range<2>(
        sycl::range<2>(1, outer_size), sycl::range<2>(1, cell_local_size)));
    this->cell_offsets.push_back(static_cast<std::size_t>(cellx));
    return {1, this->iteration_set, this->cell_offsets};
  }
}

std::size_t
ParticleLoopBlockIterationSet::get_global_size(const std::size_t N,
                                               const std::size_t local_size) {
  const auto div_mod =
      std::div(static_cast<long long>(N), static_cast<long long>(local_size));
  const std::size_t outer_size =
      static_cast<std::size_t>(div_mod.quot + (div_mod.rem == 0 ? 0 : 1)) *
      local_size;
  return outer_size;
}

std::size_t ParticleLoopBlockIterationSet::get_local_size(
    std::size_t local_size,
    const std::size_t num_bytes_local, // this is per particle
    const std::size_t stride) {
  const std::size_t local_mem_size =
      this->sycl_target->device_limits.local_mem_size;
  const std::size_t num_bytes_per_block = stride * num_bytes_local;
  NESOASSERT(num_bytes_per_block <= local_mem_size,
             "Impossible to create a local range for this stride and local "
             "memory size.");
  const std::size_t max_num_blocks_per_workgroup =
      (num_bytes_local == 0) ? local_size
                             : local_mem_size / num_bytes_per_block;
  local_size =
      std::min(get_prev_power_of_two(max_num_blocks_per_workgroup), local_size);
  NESOASSERT(local_size * stride * num_bytes_local <= local_mem_size,
             "Failure to determine a local size for iteration set.");
  return local_size;
}

ParticleLoopBlockIterationSet::ParticleLoopBlockIterationSet(
    SYCLTargetSharedPtr sycl_target, const std::size_t ncell, int *h_npart_cell,
    int *d_npart_cell)
    : sycl_target(sycl_target), ncell(ncell), h_npart_cell(h_npart_cell),
      d_npart_cell(d_npart_cell) {}

ParticleLoopBlockIterationSet::ParticleLoopBlockIterationSet(
    std::shared_ptr<ParticleDatT<REAL>> particle_dat)
    : sycl_target(particle_dat->sycl_target), ncell(particle_dat->ncell),
      h_npart_cell(particle_dat->h_npart_cell),
      d_npart_cell(particle_dat->d_npart_cell) {}

ParticleLoopBlockIterationSet::ParticleLoopBlockIterationSet(
    std::shared_ptr<ParticleDatT<INT>> particle_dat)
    : sycl_target(particle_dat->sycl_target), ncell(particle_dat->ncell),
      h_npart_cell(particle_dat->h_npart_cell),
      d_npart_cell(particle_dat->d_npart_cell) {}

std::vector<ParticleLoopBlockHost> &ParticleLoopBlockIterationSet::get_generic(
    const std::size_t cell_start, const std::size_t cell_end, std::size_t nbin,
    std::size_t local_size, const std::size_t num_bytes_local,
    const std::size_t stride) {

  local_size = this->get_local_size(local_size, num_bytes_local, stride);
  NESOASSERT(
      local_size > 0,
      "Cannot determine a local_size based on local memory requirements.");
  this->iteration_set.clear();
  this->iteration_set_size = 0;

  // The first cell the iteration set actually needs to touch.
  std::size_t cell_startv = static_cast<std::size_t>(this->ncell);
  // The last cell + 1 the iteration set actually needs to touch.
  std::size_t cell_endv = static_cast<std::size_t>(0);
  // Compute the range of cells this loop actually needs to visit.
  for (std::size_t cellx = cell_start; cellx < cell_end; cellx++) {
    const std::size_t occupancy =
        static_cast<std::size_t>(this->h_npart_cell[cellx]);
    if (occupancy) {
      cell_startv = std::min(cell_startv, cellx);
      cell_endv = std::max(cell_endv, cellx + 1);
    }
  }

  // Create an iteration block that covers all cells up to a multiple of the
  // local size.
  std::size_t min_occupancy = std::numeric_limits<std::size_t>::max();
  for (std::size_t cellx = cell_startv; cellx < cell_endv; cellx++) {
    const std::size_t occupancy =
        static_cast<std::size_t>(this->h_npart_cell[cellx]);
    min_occupancy = std::min(min_occupancy, occupancy);
  }
  const std::size_t range_cell_count = cell_endv - cell_startv;
  if ((range_cell_count == 0) || (cell_endv <= cell_startv)) {
    return this->iteration_set;
  }
  nbin = std::min(nbin, range_cell_count);

  // Truncate to a multiple of local size and stride.
  min_occupancy /= (local_size * stride);
  min_occupancy *= (local_size * stride);
  this->iteration_set_size = min_occupancy * range_cell_count;
  if (min_occupancy > 0) {
    ParticleLoopBlockDevice block_device{cell_startv, 0, this->d_npart_cell,
                                         stride};
    this->iteration_set.emplace_back(
        block_device, false, local_size,
        this->sycl_target->device_limits.validate_nd_range(sycl::nd_range<2>(
            // min_occupancy is already a multiple of local_size and stride by
            // construction.
            sycl::range<2>(range_cell_count, min_occupancy / stride),
            sycl::range<2>(1, local_size))));
  }
  // Create the peel loops
  for (std::size_t binx = 0; binx < nbin; binx++) {
    std::size_t start, end;
    get_decomp_1d(nbin, range_cell_count, binx, &start, &end);
    start += cell_startv;
    end += cell_startv;
    const std::size_t bin_width = end - start;
    std::size_t cell_max_occ = 0;
    for (std::size_t cellx = start; cellx < end; cellx++) {
      const std::size_t npart =
          static_cast<std::size_t>(this->h_npart_cell[cellx]);
      this->iteration_set_size += npart;
      cell_max_occ = std::max(cell_max_occ, npart);
    }
    // Subtract of the block already completed as this is a peel loop.
    cell_max_occ -= min_occupancy;
    if (cell_max_occ > 0) {
      const std::size_t global_range = this->get_global_size(
          get_next_multiple(cell_max_occ, stride) / stride, local_size);
      ParticleLoopBlockDevice block_device{start, min_occupancy / stride,
                                           this->d_npart_cell, stride};
      this->iteration_set.emplace_back(
          block_device, true, local_size,
          this->sycl_target->device_limits.validate_nd_range(
              sycl::nd_range<2>(sycl::range<2>(bin_width, global_range),
                                sycl::range<2>(1, local_size))));
    }
  }

  return this->iteration_set;
}

std::vector<ParticleLoopBlockHost> &
ParticleLoopBlockIterationSet::get_all_cells(std::size_t nbin,
                                             std::size_t local_size,
                                             const std::size_t num_bytes_local,
                                             const std::size_t stride) {
  return this->get_generic(0, this->ncell, nbin, local_size, num_bytes_local,
                           stride);
}

std::vector<ParticleLoopBlockHost> &
ParticleLoopBlockIterationSet::get_single_cell(
    const std::size_t cell, std::size_t local_size,
    const std::size_t num_bytes_local, const std::size_t stride) {
  return this->get_generic(cell, cell + 1, 1, local_size, num_bytes_local,
                           stride);
}

std::vector<ParticleLoopBlockHost> &
ParticleLoopBlockIterationSet::get_range_cell(const std::size_t cell_start,
                                              const std::size_t cell_end,
                                              std::size_t local_size,
                                              const std::size_t num_bytes_local,
                                              const std::size_t stride) {
  const std::size_t nbin =
      this->sycl_target->parameters->template get<SizeTParameter>("LOOP_NBIN")
          ->value;
  return this->get_generic(cell_start, cell_end, nbin, local_size,
                           num_bytes_local, stride);
}

} // namespace ParticleLoopImplementation
} // namespace NESO::Particles
