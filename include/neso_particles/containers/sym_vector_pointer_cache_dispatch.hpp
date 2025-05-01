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
   * Empty the cache of any keys that include emphemeral dats.
   */
  inline void reset_ephemeral() {
    this->cache_int.reset_ephemeral();
    this->cache_real.reset_ephemeral();
  }

  /**
   *  Create instance from map from Syms to ParticleDats
   *
   *  @param sycl_target Compute device for all ParticleDats and buffers.
   *  @param particle_dats_map_int Map from Syms to ParticleDats.
   *  @param particle_dats_map_int_eph Map from Syms to ParticleDats for
   * EphemeralDats.
   *  @param particle_dats_map_real Map from Syms to ParticleDats.
   *  @param particle_dats_map_real_eph Map from Syms to ParticleDats for
   * EphemeralDats.
   */
  SymVectorPointerCacheDispatch(
      SYCLTargetSharedPtr sycl_target,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_map_int,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_map_int_eph,
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_map_real,
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>>
          *particle_dats_map_real_eph)
      : cache_int(sycl_target, particle_dats_map_int,
                  particle_dats_map_int_eph),
        cache_real(sycl_target, particle_dats_map_real,
                   particle_dats_map_real_eph) {}

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  inline void create(std::vector<Sym<INT>> &syms) {
    this->cache_int.create(syms);
  }

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  inline void create(std::vector<Sym<REAL>> &syms) {
    this->cache_real.create(syms);
  }

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  inline void create_const(std::vector<Sym<INT>> &syms) {
    this->cache_int.create_const(syms);
  }

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  inline void create_const(std::vector<Sym<REAL>> &syms) {
    this->cache_real.create_const(syms);
  }

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

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  inline ParticleDatImplGetT<INT> get(Sym<INT> sym) {
    return this->cache_int.get(sym);
  }

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  inline ParticleDatImplGetT<REAL> get(Sym<REAL> sym) {
    return this->cache_real.get(sym);
  }

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  inline ParticleDatImplGetConstT<INT> get_const(Sym<INT> sym) {
    return this->cache_int.get_const(sym);
  }

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  inline ParticleDatImplGetConstT<REAL> get_const(Sym<REAL> sym) {
    return this->cache_real.get_const(sym);
  }
};

using SymVectorPointerCacheDispatchSharedPtr =
    std::shared_ptr<SymVectorPointerCacheDispatch>;

/**
 * Helper function to return the EphemeralDats SymVectorPointerCacheDispatch or
 * the ParticleGroup SymVectorPointerCacheDispatch.
 *
 * @param particle_group Pointer to ParticleGroup.
 * @param particle_sub_group Pointer to ParticleSubGroup, possibly nullptr.
 * @returns SymVectorPointerCacheDispatch from particle_sub_group EphemeralDats
 * if particle_sub_group is not a nullptr otherwise returns th
 * SymVectorPointerCacheDispatch from the ParticleGroup.
 */
inline SymVectorPointerCacheDispatchSharedPtr
get_sym_vector_cache_dispatch(ParticleGroup *particle_group,
                              ParticleSubGroup *particle_sub_group);

} // namespace NESO::Particles
#endif
