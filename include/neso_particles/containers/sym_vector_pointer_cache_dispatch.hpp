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
  void reset();

  /**
   * Empty the cache of any keys that include emphemeral dats.
   */
  void reset_ephemeral();

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
          *particle_dats_map_real_eph);

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  void create(std::vector<Sym<INT>> &syms);

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  void create(std::vector<Sym<REAL>> &syms);

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  void create_const(std::vector<Sym<INT>> &syms);

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Vector of Syms to create entry for.
   */
  void create_const(std::vector<Sym<REAL>> &syms);

  /**
   * Get the device pointers that correspond to the ParticleDats requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  ParticleDatImplGetT<INT> *get(std::vector<Sym<INT>> &syms);

  /**
   * Get the device pointers that correspond to the ParticleDats requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  ParticleDatImplGetT<REAL> *get(std::vector<Sym<REAL>> &syms);

  /**
   * Get the const device pointers that correspond to the ParticleDats
   * requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  ParticleDatImplGetConstT<INT> *get_const(std::vector<Sym<INT>> &syms);

  /**
   * Get the const device pointers that correspond to the ParticleDats
   * requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  ParticleDatImplGetConstT<REAL> *get_const(std::vector<Sym<REAL>> &syms);

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  ParticleDatImplGetT<INT> get(Sym<INT> sym);

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  ParticleDatImplGetT<REAL> get(Sym<REAL> sym);

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  ParticleDatImplGetConstT<INT> get_const(Sym<INT> sym);

  /**
   * Get the device pointer for a single dat.
   *
   * @param sym Sym to get device pointers for.
   * @returns Device pointer for requested dat.
   */
  ParticleDatImplGetConstT<REAL> get_const(Sym<REAL> sym);
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
