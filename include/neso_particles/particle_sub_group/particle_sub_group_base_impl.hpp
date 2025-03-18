#ifndef _NESO_PARTICLES_SUB_GROUP_PARTICLE_SUB_GROUP_BASE_IMPL_HPP_
#define _NESO_PARTICLES_SUB_GROUP_PARTICLE_SUB_GROUP_BASE_IMPL_HPP_

#include "particle_loop_sub_group.hpp"
#include "particle_sub_group_base.hpp"

namespace NESO::Particles {

inline void ParticleSubGroup::get_cells_layers(INT *d_cells, INT *d_layers) {

  auto lambda_loop = [&](auto iteration_set) {
    particle_loop(
        iteration_set,
        [=](auto index) {
          const INT px = index.get_loop_linear_index();
          d_cells[px] = index.cell;
          d_layers[px] = index.layer;
        },
        Access::read(ParticleLoopIndex{}))
        ->execute();
  };

  if (this->is_entire_particle_group()) {
    lambda_loop(this->particle_group);
  } else {
    lambda_loop(std::shared_ptr<ParticleSubGroup>(
        this, []([[maybe_unused]] auto x) {}));
  }
}

} // namespace NESO::Particles

#endif
