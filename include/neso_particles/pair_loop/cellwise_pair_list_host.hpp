#ifndef __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_HOST_HPP_
#define __NESO_PARTICLES_PAIR_LOOP_DSMC_CELLWISE_PAIR_LIST_HOST_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../device_buffers.hpp"

#include <map>
#include <tuple>
#include <vector>

namespace NESO::Particles {

using CellwisePairListHostMap =
    std::map<int,          // The first key is the cell index
             std::map<int, // The second key is the wave index
                      std::pair<
                          // The "values" are a pair where the first entry is
                          // the i particles.
                          std::vector<int>,
                          // The second entry is the j particles.
                          std::vector<int>>>>;

/**
 * Type to hold a list of pairs of particles that should collide in each cell.
 */
class CellwisePairListHost {
protected:
  // Map from particles to the current wave.
  std::map<std::tuple<int, int>, int> map_particles_to_wave;

  // Map from wave number to pairs of particles
  CellwisePairListHostMap map_wave_to_pairs;

public:
  /// Number of cells to hold pairs for.
  int cell_count{0};

  /// Disable (implicit) copies.
  CellwisePairListHost(const CellwisePairListHost &st) = delete;
  /// Disable (implicit) copies.
  CellwisePairListHost &operator=(CellwisePairListHost const &a) = delete;
  ~CellwisePairListHost() = default;

  /**
   * @param cell_count Number of cells to hold pairs for.
   */
  CellwisePairListHost(const int cell_count);

  /**
   * Adds a pair of particles to the pair list for a given cell.
   */
  void push_back(const int cell, const int i, const int j);

  /**
   * @param cell Cell of particles.
   * @param layer Layer of particle.
   * @returns Minimum wave index that a pair involving the given particle could
   * execute on.
   */
  int get_next_wave(const int cell, const int layer);

  /**
   * @param cell Cell of particles.
   * @param layer Layer of particle.
   * @param wave Next wave that can include the particle.
   */
  void set_next_wave(const int cell, const int layer, const int wave);

  /**
   * Clears all held state.
   */
  void clear();

  /**
   * @returns The host representation of the pair list.
   */
  CellwisePairListHostMap &get();
};

using CellwisePairListHostSharedPtr = std::shared_ptr<CellwisePairListHost>;

} // namespace NESO::Particles

#endif
