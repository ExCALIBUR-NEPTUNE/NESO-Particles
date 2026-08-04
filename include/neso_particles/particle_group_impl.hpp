#ifndef _NESO_PARTICLES_PARTICLE_GROUP_IMPL_H_
#define _NESO_PARTICLES_PARTICLE_GROUP_IMPL_H_

#include "containers/descendant_products.hpp"
#include "global_mapping.hpp"
#include "loop/particle_loop_iteration_set.hpp"
#include "particle_group.hpp"
#include "particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

inline void ParticleGroup::add_particles() {
  NESOASSERT(false, "Not implemented yet - use add_particles_local and hybrid "
                    "move or parallel advection initialisation.");
};
template <typename U>
inline void ParticleGroup::add_particles([[maybe_unused]] U particle_data) {
  NESOASSERT(false, "Not implemented yet - use add_particles_local and hybrid "
                    "move or parallel advection initialisation.");
};

template <typename T>
inline void ParticleGroup::remove_particles(const int npart, T *usm_cells,
                                            T *usm_layers) {
  this->layer_compressor.remove_particles(npart, usm_cells, usm_layers);
  this->set_npart_cell_from_dat();
  this->invalidate_group_version();
}

template <typename... T>
inline void ParticleGroup::print(std::ostream &os, T &&...args) {
  SymStore print_spec(std::forward<T>(args)...);
  this->print_inner(os, print_spec);
}

template <typename... T>
inline void ParticleGroup::print(std::ofstream &os, T &&...args) {
  SymStore print_spec(std::forward<T>(args)...);
  this->print_inner(os, print_spec);
}

template <typename... T> inline void ParticleGroup::print(T &&...args) {
  SymStore print_spec(std::forward<T>(args)...);
  this->print_inner(std::cout, print_spec);
}

} // namespace NESO::Particles

#endif
