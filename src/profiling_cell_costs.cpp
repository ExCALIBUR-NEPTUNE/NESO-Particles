#include <neso_particles/profiling_cell_costs.hpp>

namespace NESO::Particles {

ProfilingCellCosts::ProfilingCellCosts(const int num_cells)
    : num_cells(num_cells) {

  const bool enable_tmp =
      static_cast<bool>(get_env_size_t("NESO_PARTICLES_CELL_COST_PROFILE", 0));

  this->enabled = false;
  if (enable_tmp) {
    this->enable();
  }
}

void ProfilingCellCosts::enable() {
  if (!this->enabled) {
    this->enabled = true;
    this->h_cell_costs = std::vector<REAL>(num_cells);
    this->h_npart_cell = std::vector<int>(num_cells);
    this->zero();
  }
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

  this->accumulate(sycl_target, 0, this->num_cells, d_npart_cell,
                   per_particle_cost);
}

void ProfilingCellCosts::accumulate(SYCLTargetSharedPtr sycl_target,
                                    const int cell_start, const int cell_end,
                                    const int *RESTRICT d_npart_cell,
                                    const REAL per_particle_cost) {

  if (this->enabled) {
    NESOASSERT(cell_start <= cell_end, "cell_start > cell_end");
    NESOASSERT((0 <= cell_start) && (cell_start <= this->num_cells),
               "Require 0<=cell_start<=num_cells");
    NESOASSERT((0 <= cell_end) && (cell_end <= this->num_cells),
               "Require 0<=cell_end<=num_cells");

    const int num_cells_copy = cell_end - cell_start;

    sycl_target
        ->memcpy(this->h_npart_cell.data() + cell_start,
                 d_npart_cell + cell_start, num_cells_copy * sizeof(int))
        .wait_and_throw();

    for (int ix = cell_start; ix < cell_end; ix++) {
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
  } else {
    cell_costs.clear();
  }
}

} // namespace NESO::Particles
