#ifndef _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_DISPATCH_H_
#define _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_DISPATCH_H_

#include "sym_vector_pointer_cache.hpp"

namespace NESO::Particles {

/**
 * Wrapper around SymVectorPointerCache for REAL and INT Syms.
 */
class SymVectorPointerCacheDispatch {
public:
  /// The underlying cache for INT pointers.
  SymVectorPointerCache<INT> cache_int;
  /// The underlying cache for REAL pointers.
  SymVectorPointerCache<REAL> cache_real;

  /**
   * Reset the container and empty the cache.
   */
  inline void reset() {
    this->cache_int.reset();
    this->cache_real.reset();
  }

  /**
   *  Create instance from map from Syms to ParticleDats
   *
   *  @param sycl_target Compute device for all ParticleDats and buffers.
   *  @param particle_dats_map Map from Syms to ParticleDats.
   */
  SymVectorPointerCacheDispatch(
      SYCLTargetSharedPtr sycl_target,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_map_int,
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_map_real)
      : cache_int(sycl_target, particle_dats_map_int),
        cache_real(sycl_target, particle_dats_map_real) {}

  /**
   * Get the device pointers that correspond to the ParticleDats requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetT<INT> *get(std::vector<Sym<INT>> &syms) {
    return this->cache_int.get(syms);
  }

  /**
   * Get the device pointers that correspond to the ParticleDats requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetT<REAL> *get(std::vector<Sym<REAL>> &syms) {
    return this->cache_real.get(syms);
  }

  /**
   * Get the const device pointers that correspond to the ParticleDats
   * requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetConstT<INT> *get_const(std::vector<Sym<INT>> &syms) {
    return this->cache_int.get_const(syms);
  }

  /**
   * Get the const device pointers that correspond to the ParticleDats
   * requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetConstT<REAL> *
  get_const(std::vector<Sym<REAL>> &syms) {
    return this->cache_real.get_const(syms);
  }
};

} // namespace NESO::Particles
#endif
