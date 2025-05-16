#include <neso_particles/common_impl.hpp>
#include <neso_particles/particle_group_temporary.hpp>

namespace NESO::Particles {

ParticleGroupSharedPtr
ParticleGroupTemporary::get(const ParticleGroupSharedPtr &particle_group) {
  if (particle_group->resource_stack_particle_group_temporary == nullptr) {

    auto tmp = std::make_shared<Private::ParticleGroupTemporaryRSI>();
    tmp->domain = particle_group->domain;
    tmp->particle_spec = &particle_group->particle_spec;
    tmp->sycl_target = particle_group->sycl_target;
    particle_group->resource_stack_particle_group_temporary =
        std::make_shared<ResourceStack<ParticleGroup>>(tmp);
  }

  auto ptr = std::dynamic_pointer_cast<ResourceStack<ParticleGroup>>(
      particle_group->resource_stack_particle_group_temporary);
  NESOASSERT(ptr != nullptr, "Could not cast ptr.");

  auto tmp_particle_group = ptr->get();

  auto lambda_add_dat = [&](auto sym, const int ncomp) {
    tmp_particle_group->add_particle_dat(sym, ncomp);
  };

  auto lambda_check_container_new = [&](auto m) {
    for (auto &[sym, dat] : m) {
      const int ncomp = dat->ncomp;
      if (tmp_particle_group->contains_dat(sym)) {
        auto tmp_dat = tmp_particle_group->get_dat(sym);
        const int tmp_ncomp = tmp_dat->ncomp;
        if (ncomp != tmp_ncomp) {
          tmp_particle_group->remove_particle_dat(sym);
          lambda_add_dat(sym, ncomp);
        }
      } else {
        lambda_add_dat(sym, ncomp);
      }
    }
  };

  lambda_check_container_new(particle_group->particle_dats_real);
  lambda_check_container_new(particle_group->particle_dats_int);

  auto lambda_check_container_old = [&](auto m) {
    for (auto &[sym, dat] : m) {
      if (!particle_group->contains_dat(sym)) {
        tmp_particle_group->remove_particle_dat(sym);
      }
    }
  };

  lambda_check_container_old(tmp_particle_group->particle_dats_real);
  lambda_check_container_old(tmp_particle_group->particle_dats_int);

  return tmp_particle_group;
}

void ParticleGroupTemporary::restore(
    const ParticleGroupSharedPtr &particle_group,
    ParticleGroupSharedPtr &temporary_particle_group) {
  auto ptr = std::dynamic_pointer_cast<ResourceStack<ParticleGroup>>(
      particle_group->resource_stack_particle_group_temporary);
  NESOASSERT(ptr != nullptr, "Could not cast ptr.");
  ptr->restore(temporary_particle_group);
}

} // namespace NESO::Particles
