#ifndef _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_
#define _NESO_PARTICLES_PARTICLE_LOOP_ITERATION_SET_HPP_

#include "../sycl_typedefs.hpp"
#include "../typedefs.hpp"

namespace NESO::Particles::ParticleLoopImplementation {

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

  /**
   *  Creates iteration set creator for a given set of cell particle counts.
   *
   *  @param nbin Number of blocks of cells.
   *  @param ncell Number of cells.
   *  @param h_npart_cell Host accessible array of cell particle counts.
   */
  ParticleLoopIterationSet(const int nbin, const int ncell, int *h_npart_cell)
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
