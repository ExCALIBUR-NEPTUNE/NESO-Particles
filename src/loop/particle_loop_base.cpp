#include <neso_particles/loop/particle_loop_base.hpp>
#include <neso_particles/particle_dat.hpp>
#include <neso_particles/particle_group.hpp>

namespace NESO::Particles {

bool determine_iteration_set(const int ncell,
                             const std::optional<int> cell_start,
                             const std::optional<int> cell_end,
                             int *cell_start_v, int *cell_end_v) {
  if ((cell_start == std::nullopt) && (cell_end == std::nullopt)) {
    // Is all cells
    *cell_start_v = 0;
    *cell_end_v = ncell;
    return true;
  } else if ((cell_start != std::nullopt) && (cell_end == std::nullopt)) {
    // Is single cell
    *cell_start_v = cell_start.value();
    *cell_end_v = *cell_start_v + 1;
    return false;
  } else if ((cell_start != std::nullopt) && (cell_end != std::nullopt)) {
    // Is provided cell range.
    *cell_start_v = cell_start.value();
    *cell_end_v = cell_end.value();
    const bool default_start = cell_start.value() == 0;
    const bool default_end = cell_end.value() == ncell;
    return default_start && default_end;
  } else {
    NESOASSERT(false, "Bad cell_start, cell_end found.");
    return false;
  }
}

void ParticleLoopBase::init_from_particle_dat(
    std::shared_ptr<ParticleDatT<REAL>> particle_dat) {
  this->ncell = particle_dat->ncell;
  this->h_npart_cell_lb = particle_dat->h_npart_cell;
  this->d_npart_cell = particle_dat->d_npart_cell;
  this->d_npart_cell_lb = this->d_npart_cell;
  this->d_npart_cell_es = particle_dat->get_d_npart_cell_es();
  this->d_npart_cell_es_lb = this->d_npart_cell_es;
  this->iteration_set = std::make_unique<
      ParticleLoopImplementation::ParticleLoopBlockIterationSet>(particle_dat);
}

void ParticleLoopBase::init_from_particle_dat(
    std::shared_ptr<ParticleDatT<INT>> particle_dat) {
  this->ncell = particle_dat->ncell;
  this->h_npart_cell_lb = particle_dat->h_npart_cell;
  this->d_npart_cell = particle_dat->d_npart_cell;
  this->d_npart_cell_lb = this->d_npart_cell;
  this->d_npart_cell_es = particle_dat->get_d_npart_cell_es();
  this->d_npart_cell_es_lb = this->d_npart_cell_es;
  this->iteration_set = std::make_unique<
      ParticleLoopImplementation::ParticleLoopBlockIterationSet>(particle_dat);
}

bool ParticleLoopBase::iteration_set_is_empty(
    const std::optional<int> cell_start, const std::optional<int> cell_end) {

  int cell_start_v = -1;
  int cell_end_v = -1;
  const bool all_cells = determine_iteration_set(
      this->ncell, cell_start, cell_end, &cell_start_v, &cell_end_v);

  if (!all_cells) {
    NESOASSERT(
        (cell_start_v > -1) && (cell_start_v <= this->ncell) &&
            (cell_end_v > -1) && (cell_end_v <= this->ncell),
        "ParticleLoop execute or submit called on cell that does not exist.");
    int max_npart = 0;
    for (int cellx = cell_start_v; cellx < cell_end_v; cellx++) {
      max_npart = std::max(max_npart, this->h_npart_cell_lb[cellx]);
    }
    return max_npart == 0;
  } else if (this->particle_group_ptr != nullptr) {
    return this->particle_group_ptr->get_npart_local() == 0;
  } else {
    return false;
  }
}

void ParticleLoopBase::profiling_region_init() {
  this->profile_region = ProfileRegion(this->loop_type, this->name);
}

void ParticleLoopBase::profile_region_finalise() {
  this->profile_region.end();
  this->sycl_target->profile_map.add_region(this->profile_region);
}

} // namespace NESO::Particles
