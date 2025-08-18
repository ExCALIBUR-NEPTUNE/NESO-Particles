#include <neso_particles/loop/pli_particle_dat.hpp>

namespace NESO::Particles {
namespace ParticleLoopImplementation {

/**
 * Method to compute access to a particle dat (read).
 */
ParticleDatImplGetConstT<REAL>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ParticleDatT<REAL> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a particle dat (write).
 */
ParticleDatImplGetT<REAL>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ParticleDatT<REAL> *> &a) {
  return a.obj->impl_get();
}

/**
 * Method to compute access to a particle dat (read).
 */
ParticleDatImplGetConstT<INT>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ParticleDatT<INT> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a particle dat (write).
 */
ParticleDatImplGetT<INT>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ParticleDatT<INT> *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation
} // namespace NESO::Particles
