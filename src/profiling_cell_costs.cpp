#include <neso_particles/profiling_cell_costs.hpp>

namespace NESO::Particles {

ProfilingCellCosts::ProfilingCellCosts(SYCLTargetSharedPtr sycl_target,
                                       const int num_cells)
    : sycl_target(sycl_target), num_cells(num_cells) {

  this->enabled =
      static_cast<bool>(get_env_size_t("NESO_PARTICLES_CELL_COST_PROFILE", 0));

  if (this->enabled) {
    this->d_cell_costs =
        std::make_shared<BufferDevice<REAL>>(sycl_target, num_cells);
  }

  this->zero();
}

void ProfilingCellCosts::zero() {
  if (this->enabled) {
    this->sycl_target->queue
        .fill<REAL>(this->d_cell_costs->ptr, 0.0,
                    this->d_cell_costs->size_bytes())
        .wait_and_throw();
  }
}

void ProfilingCellCosts::accumulate(const REAL *RESTRICT d_npart_cell,
                                    const REAL per_particle_cost) {

  REAL *RESTRICT k_cell_costs = this->d_cell_costs->ptr;
  this->sycl_target->queue
      .parallel_for(sycl::range<1>(this->num_cells),
                    [=](auto ix) {
                      k_cell_costs[ix] += d_npart_cell[ix] * per_particle_cost;
                    })
      .wait_and_throw();
}

} // namespace NESO::Particles
