#include <neso_particles/profiling_cell_costs.hpp>

namespace NESO::Particles {

ProfilingCellCosts::ProfilingCellCosts(const int num_cells)
    : num_cells(num_cells) {

  this->enabled =
      static_cast<bool>(get_env_size_t("NESO_PARTICLES_CELL_COST_PROFILE", 0));

  if (this->enabled) {
    this->enable();
  }

  this->zero();
}

void ProfilingCellCosts::enable() {
  this->enabled = true;
  this->h_cell_costs = std::vector<REAL>(num_cells);
  this->h_npart_cell = std::vector<int>(num_cells);
}

void ProfilingCellCosts::disable() {
  this->enabled = false;
  this->h_cell_costs.clear();
  this->h_npart_cell.clear();
}

void ProfilingCellCosts::zero() {
  if (this->enabled) {
    std::fill(this->h_cell_costs.begin(), this->h_cell_costs.end(), 0.0);
  }
}

void ProfilingCellCosts::accumulate(SYCLTargetSharedPtr sycl_target,
                                    const int *RESTRICT d_npart_cell,
                                    const REAL per_particle_cost) {

  if (this->enabled) {
    sycl_target->queue
        .memcpy(this->h_npart_cell.data(), d_npart_cell,
                this->num_cells * sizeof(int))
        .wait_and_throw();

    for (int ix = 0; ix < this->num_cells; ix++) {
      this->h_cell_costs[ix] += per_particle_cost * this->h_npart_cell[ix];
    }
  }
}

void ProfilingCellCosts::get_cell_costs(std::vector<REAL> &cell_costs) {

  if (this->enabled) {
    if (cell_costs.size() < this->h_cell_costs.size()) {
      cell_costs.resize(this->h_cell_costs.size());
    }

    std::memcpy(cell_costs.data(), this->h_cell_costs.data(),
                this->h_cell_costs.size() * sizeof(REAL));
  }
}

} // namespace NESO::Particles
