#include <neso_particles/pair_loop/cellwise_pair_list_host.hpp>

namespace NESO::Particles {

CellwisePairListHost::CellwisePairListHost(const int cell_count)
    : cell_count(cell_count) {}

void CellwisePairListHost::push_back(const int cell, const int i, const int j) {
  NESOASSERT((0 <= cell) && (cell < this->cell_count), "Bad cell index.");
  const int wave =
      std::max(this->get_next_wave(cell, i), this->get_next_wave(cell, j));
  const int oi = i < j ? i : j;
  const int oj = i < j ? j : i;
  this->map_wave_to_pairs[cell][wave].first.push_back(oi);
  this->map_wave_to_pairs[cell][wave].second.push_back(oj);
  this->set_next_wave(cell, i, wave + 1);
  this->set_next_wave(cell, j, wave + 1);
}

int CellwisePairListHost::get_next_wave(const int cell, const int layer) {
  NESOASSERT((0 <= cell) && (cell < this->cell_count), "Bad cell index.");
  if (this->map_particles_to_wave.count({cell, layer})) {
    return this->map_particles_to_wave[{cell, layer}];
  } else {
    return 0;
  }
}

void CellwisePairListHost::set_next_wave(const int cell, const int layer,
                                         const int wave) {
  NESOASSERT((0 <= cell) && (cell < this->cell_count), "Bad cell index.");
  NESOASSERT((0 <= wave) && (wave <= (this->map_wave_to_pairs[cell].size())),
             "Next wave cannot be set beyond the current max wave + 1.");
  this->map_particles_to_wave[{cell, layer}] = wave;
}

void CellwisePairListHost::clear() {
  this->map_particles_to_wave.clear();
  this->map_wave_to_pairs.clear();
}

CellwisePairListHostMap &CellwisePairListHost::get() {
  return this->map_wave_to_pairs;
}

} // namespace NESO::Particles
