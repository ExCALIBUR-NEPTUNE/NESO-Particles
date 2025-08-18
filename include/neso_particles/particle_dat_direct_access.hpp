#ifndef _NESO_PARTICLES_PARTICLE_DAT_DIRECT_ACCESS_HPP_
#define _NESO_PARTICLES_PARTICLE_DAT_DIRECT_ACCESS_HPP_

#include "loop/access_descriptors.hpp"
#include "particle_dat.hpp"

namespace NESO::Particles {

namespace Access {

/**
 * This file contains functions for accessing ParticleDat data directly, e.g. in
 * custom SYCL loops. These functions allow this access to happen in a manner
 * which composes with the particle data tracking used by other parts of NP,
 * e.g. ParticleSubGroups.
 */

/**
 * Get read access to particle data in a ParticleDat. The corresponding call to
 * direct_restore must be made.
 *
 * @param dat_access ParticleDat to access wrapped in a Access::read call.
 * @returns Device pointer to cells, components and layers.
 */
template <typename T>
[[nodiscard]] inline ParticleDatImplGetConstT<T>
direct_get(Read<ParticleDatSharedPtr<T>> dat_access) {
  return dat_access.obj->impl_get_const();
}

/**
 * Get write access to particle data in a ParticleDat. The corresponding call to
 * direct_restore must be made.
 *
 * @param dat_access ParticleDat to access wrapped in a Access::write call.
 * @returns Device pointer to cells, components and layers.
 */
template <typename T>
[[nodiscard]] inline ParticleDatImplGetT<T>
direct_get(Write<ParticleDatSharedPtr<T>> dat_access) {
  return dat_access.obj->impl_get();
}

/**
 * Restore read access to particle data in a ParticleDat. The corresponding call
 * to direct_get must be made prior to this function call.
 *
 * @param[in] dat_access ParticleDat which was accessed wrapped in a
 * Access::read call.
 * @param[in, out] Device pointer which was returned from call to direct_get.
 * This pointer will be set to nullptr on return.
 */
template <typename T>
inline void direct_restore(Read<ParticleDatSharedPtr<T>> dat_access,
                           ParticleDatImplGetConstT<T> &data) {
  NESOASSERT(dat_access.obj->impl_get_const() == data,
             "Passed pointer is different to that given by direct_get.");
  data = nullptr;
}

/**
 * Restore write access to particle data in a ParticleDat. The corresponding
 * call to direct_get must be made prior to this function call.
 *
 * @param[in] dat_access ParticleDat which was accessed wrapped in a
 * Access::write call.
 * @param[in, out] Device pointer which was returned from call to direct_get.
 * This pointer will be set to nullptr on return.
 */
template <typename T>
inline void direct_restore(Write<ParticleDatSharedPtr<T>> dat_access,
                           ParticleDatImplGetT<T> &data) {
  NESOASSERT(dat_access.obj->check_ptr_is_same(data),
             "Passed pointer is different to that given by direct_get.");
  data = nullptr;
}

} // namespace Access
} // namespace NESO::Particles

#endif
