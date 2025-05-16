#include <neso_particles/cell_dat_compression.hpp>
#include <neso_particles/departing_particle_identification_impl.hpp>
#include <neso_particles/loop/particle_loop_impl.hpp>

namespace NESO::Particles {

void LayerCompressor::remove_particles(const int npart, int *usm_cells,
                                       int *usm_layers) {
  this->remove_particles_inner(npart, usm_cells, usm_layers);
}

void LayerCompressor::remove_particles(const int npart, INT *usm_cells,
                                       INT *usm_layers) {
  this->remove_particles_inner(npart, usm_cells, usm_layers);
}

} // namespace NESO::Particles
