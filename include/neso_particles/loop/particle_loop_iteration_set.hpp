#ifndef _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_
#define _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_

#include "../sycl_typedefs.hpp"
#include "../typedefs.hpp"

namespace NESO::Particles::ParticleLoopImplementation {

struct ParticleLoopNDRangeSet {
  int layer_offset;
  int bin_start;
  int bin_end;
  std::vector<sycl::nd_range<2>> &nd_ranges;
  std::vector<std::size_t> &cell_offsets;
};

/**
 * For a set of cells containing particles create several sycl::nd_range
 * instances which cover the iteration space of all particles. This exists to
 * create an iteration set over all particles which is blocked, to reduce the
 * number of kernel launches, and reasonably robust to non-uniform.
 */
struct ParticleLoopIterationSet {

  /// The number of blocks of cells.
  const int nbin;
  /// The number of cells.
  const int ncell;
  /// Host accessible pointer to the number of particles in each cell.
  int *h_npart_cell;
  /// Container to store the sycl::nd_ranges.
  std::vector<sycl::nd_range<2>> iteration_set;
  /// Offsets to add to the cell index to map to the correct cell.
  std::vector<std::size_t> cell_offsets;

  std::vector<int> bin_maxes;
  std::vector<int> bin_avgs;
  std::vector<int> bin_cell_starts;
  std::vector<int> bin_cell_ends;

  /**
   *  Creates iteration set creator for a given set of cell particle counts.
   *
   *  @param nbin Number of blocks of cells.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   */
  ParticleLoopIterationSet(const int nbin, const int ncell, int *h_npart_cell)
      : nbin(std::min(ncell, nbin)), ncell(ncell), h_npart_cell(h_npart_cell) {
    this->iteration_set.reserve(nbin + 1);
    this->cell_offsets.reserve(nbin + 1);
    this->bin_maxes.reserve(nbin);
    this->bin_avgs.reserve(nbin);
    this->bin_cell_starts.reserve(nbin);
    this->bin_cell_ends.reserve(nbin);
  }

  /**
   *  Create and return an iteration set which is formed as nbin
   *  sycl::nd_ranges.
   *
   *  @param cell If set iteration set will only cover this cell.
   *  @param local_size Optional size of SYCL work groups.
   *  @returns Struct containing iteration set ranges.
   */
  inline ParticleLoopNDRangeSet
  get(const std::optional<int> cell = std::nullopt,
      const size_t local_size = 256) {

    this->iteration_set.clear();
    this->cell_offsets.clear();
    this->bin_maxes.clear();
    this->bin_avgs.clear();
    this->bin_cell_starts.clear();
    this->bin_cell_ends.clear();

    if (cell == std::nullopt) {
      int min_occupancy = std::numeric_limits<int>::max();
      for (int binx = 0; binx < nbin; binx++) {
        int start, end;
        get_decomp_1d(nbin, ncell, binx, &start, &end);
        int cell_maxi = 0;
        int cell_avg = 0;
        for (int cellx = start; cellx < end; cellx++) {
          const int cell_occ = h_npart_cell[cellx];
          cell_maxi = std::max(cell_maxi, cell_occ);
          cell_avg += cell_occ;
          min_occupancy = std::min(cell_occ, min_occupancy);
        }
        cell_avg = (((REAL)cell_avg) / ((REAL)(end - start)));
        this->bin_maxes.push_back(cell_maxi);
        this->bin_avgs.push_back(cell_avg);
        this->bin_cell_starts.push_back(start);
        this->bin_cell_ends.push_back(end);
      }

      int layer_offset = 0;
      int bin_start = 0;
      int bin_end = 0;
      if (min_occupancy > 0) {
        const std::size_t cell_local_size =
            get_min_power_of_two((std::size_t)min_occupancy, local_size);
        const auto div_mod = std::div(static_cast<long long>(min_occupancy),
                                      static_cast<long long>(cell_local_size));
        const std::size_t outer_size =
            static_cast<std::size_t>(div_mod.quot) * cell_local_size;
        layer_offset = static_cast<int>(outer_size);
        this->iteration_set.emplace_back(
            sycl::nd_range<2>(sycl::range<2>(this->ncell, outer_size),
                              sycl::range<2>(1, cell_local_size)));
        this->cell_offsets.push_back(static_cast<std::size_t>(0));
        bin_start = 1;
        bin_end += 1;
      }
      for (int binx = 0; binx < nbin; binx++) {
        const int start = this->bin_cell_starts.at(binx);
        const int end = this->bin_cell_ends.at(binx);
        const int bin_width = end - start;
        const int cell_maxi = this->bin_maxes.at(binx) - layer_offset;
        const int cell_avg =
            std::max(this->bin_avgs.at(binx) - layer_offset, 1);

        const std::size_t cell_local_size =
            get_min_power_of_two((std::size_t)cell_avg, local_size);
        const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                      static_cast<long long>(cell_local_size));
        const std::size_t outer_size =
            static_cast<std::size_t>(div_mod.quot +
                                     (div_mod.rem == 0 ? 0 : 1)) *
            cell_local_size;

        if (cell_maxi > 0) {
          this->iteration_set.emplace_back(
              sycl::nd_range<2>(sycl::range<2>(bin_width, outer_size),
                                sycl::range<2>(1, cell_local_size)));
          this->cell_offsets.push_back(static_cast<std::size_t>(start));
          bin_end++;
        }
      }

      return {layer_offset, bin_start, bin_end, this->iteration_set,
              this->cell_offsets};
    } else {
      int bin_start = 1;
      int bin_end = 1;
      const int cellx = cell.value();
      const size_t cell_maxi = static_cast<size_t>(h_npart_cell[cellx]);
      const size_t cell_local_size =
          get_min_power_of_two((size_t)cell_maxi, local_size);

      const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                    static_cast<long long>(cell_local_size));
      const std::size_t outer_size0 =
          static_cast<std::size_t>(div_mod.quot) * cell_local_size;
      const int layer_offset = static_cast<int>(outer_size0);

      this->iteration_set.emplace_back(sycl::nd_range<2>(
          sycl::range<2>(1, outer_size0), sycl::range<2>(1, cell_local_size)));
      this->cell_offsets.push_back(static_cast<std::size_t>(cellx));

      const auto remaining = cell_maxi - layer_offset;
      if (remaining) {
        this->iteration_set.emplace_back(sycl::nd_range<2>(
            sycl::range<2>(1, cell_local_size), sycl::range<2>(1, cell_local_size)));
        this->cell_offsets.push_back(static_cast<std::size_t>(cellx));
        bin_end++;
      }

      return {layer_offset, bin_start, bin_end, this->iteration_set,
              this->cell_offsets};
    }
  }
};

/**
 * For a set of cells containing particles create several sycl::nd_range
 * instances which cover the iteration space of all particles. This exists to
 * create an iteration set over all particles which is blocked, to reduce the
 * number of kernel launches, and reasonably robust to non-uniform.
 */
struct ParticleLoopIterationSetNoBlock {

  /// The number of blocks of cells.
  const int nbin;
  /// The number of cells.
  const int ncell;
  /// Host accessible pointer to the number of particles in each cell.
  int *h_npart_cell;
  /// Container to store the sycl::nd_ranges.
  std::vector<sycl::nd_range<2>> iteration_set;
  /// Offsets to add to the cell index to map to the correct cell.
  std::vector<std::size_t> cell_offsets;

  /**
   *  Creates iteration set creator for a given set of cell particle counts.
   *
   *  @param nbin Number of blocks of cells.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   */
  ParticleLoopIterationSetNoBlock(const int nbin, const int ncell,
                                  int *h_npart_cell)
      : nbin(std::min(ncell, nbin)), ncell(ncell), h_npart_cell(h_npart_cell) {
    this->iteration_set.reserve(nbin);
    this->cell_offsets.reserve(nbin);
  }

  /**
   *  Create and return an iteration set which is formed as nbin
   *  sycl::nd_ranges.
   *
   *  @param cell If set iteration set will only cover this cell.
   *  @param local_size Optional size of SYCL work groups.
   *  @returns Tuple containing: Number of bins, sycl::nd_ranges, cell index
   *  offsets.
   */
  inline std::tuple<int, std::vector<sycl::nd_range<2>> &,
                    std::vector<std::size_t> &>
  get(const std::optional<int> cell = std::nullopt,
      const size_t local_size = 256) {

    this->iteration_set.clear();
    this->cell_offsets.clear();

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
        }
        cell_avg = (((REAL)cell_avg) / ((REAL)(end - start)));
        const size_t cell_local_size =
            get_min_power_of_two((size_t)cell_avg, local_size);
        const auto div_mod = std::div(static_cast<long long>(cell_maxi),
                                      static_cast<long long>(cell_local_size));
        const std::size_t outer_size =
            static_cast<std::size_t>(div_mod.quot +
                                     (div_mod.rem == 0 ? 0 : 1)) *
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
};

} // namespace NESO::Particles::ParticleLoopImplementation

#endif
