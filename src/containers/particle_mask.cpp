#include <neso_particles/containers/particle_mask.hpp>
#include <neso_particles/containers/sym_vector_pointer_cache_dispatch_impl.hpp>
#include <neso_particles/loop/particle_loop.hpp>
#include <neso_particles/loop/particle_loop_args.hpp>
#include <neso_particles/loop/particle_loop_args_impl.hpp>
#include <neso_particles/loop/particle_loop_base.hpp>
#include <neso_particles/loop/particle_loop_functions.hpp>

namespace NESO::Particles {

namespace ParticleLoopImplementation {

Access::MaskArray::Read
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ParticleMask *> &a) {
  Access::Read<MaskArray *> b;
  b.obj = a.obj;
  return create_loop_arg(global_info, cgh, b);
}

Access::MaskArray::Write
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ParticleMask *> &a) {
  Access::Write<MaskArray *> b;
  b.obj = a.obj;
  return create_loop_arg(global_info, cgh, b);
}

} // namespace ParticleLoopImplementation

ParticleMask::ParticleMask(SYCLTargetSharedPtr sycl_target)
    : MaskArray(sycl_target, 1) {}

void ParticleMask::reset(ParticleGroupSharedPtr particle_group) {
  MaskArray::reset(false,
                   static_cast<std::size_t>(particle_group->get_npart_local()));
}

void ParticleMask::set(ParticleGroupSharedPtr particle_group, Sym<INT> sym,
                       const int component) {

  const auto npart_local = particle_group->get_npart_local();
  if (this->size != npart_local) {
    this->reset(particle_group);
  }

  const int k_component = component;
  auto k_mask_array_device = this->get_device();

  particle_loop(
      particle_group,
      [=](auto INDEX, auto DAT) {
        const bool value = DAT.at(k_component) != 0;
        k_mask_array_device.set(INDEX.get_local_linear_index(), 0, value);
      },
      Access::read(ParticleLoopIndex{}), Access::read(sym))
      ->execute();
}

void ParticleMask::set(ParticleGroupSharedPtr particle_group,
                       const bool value) {

  const auto npart_local = particle_group->get_npart_local();
  if (this->size != npart_local) {
    this->reset(particle_group);
  }

  const bool k_value = value;
  auto k_mask_array_device = this->get_device();

  particle_loop(
      particle_group,
      [=](auto INDEX) {
        k_mask_array_device.set(INDEX.get_local_linear_index(), 0, k_value);
      },
      Access::read(ParticleLoopIndex{}))
      ->execute();
}

void ParticleMask::get(ParticleGroupSharedPtr particle_group, Sym<INT> sym,
                       const int component) {

  const auto npart_local = particle_group->get_npart_local();
  NESOASSERT(this->size == npart_local,
             "Missmatch between number of masks and number of particles.");

  const int k_component = component;
  auto k_mask_array_device = this->get_device();

  particle_loop(
      particle_group,
      [=](auto INDEX, auto DAT) {
        DAT.at(k_component) =
            k_mask_array_device.get(INDEX.get_local_linear_index(), 0);
      },
      Access::read(ParticleLoopIndex{}), Access::write(sym))
      ->execute();
}

} // namespace NESO::Particles
