#ifndef _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_H_
#define _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_H_

#include "../compute_target.hpp"
#include "../particle_dat.hpp"
#include <map>
#include <memory>
#include <vector>

namespace NESO::Particles {

/**
 * This container caches the device buffers and pointers required by SymVector
 * instances such that no mallocs/copies/frees have to occur if the SymVector
 * has been created already and exists in the cache.
 */
template <typename T> class SymVectorPointerCache {
protected:
  SYCLTargetSharedPtr sycl_target;
  /// Pointer to the ParticleGroup map from Sym to ParticleDat.
  std::map<Sym<T>, ParticleDatSharedPtr<T>> *particle_dats_map;
  std::map<std::vector<Sym<T>>,
           std::unique_ptr<BufferDevice<ParticleDatImplGetT<T>>>>
      map_syms_ptrs;
  std::map<std::vector<Sym<T>>,
           std::unique_ptr<BufferDevice<ParticleDatImplGetConstT<T>>>>
      map_syms_const_ptrs;

public:
  SymVectorPointerCache<T> &
  operator=(const SymVectorPointerCache<T> &) = delete;

  /**
   * Reset the container and empty the cache.
   */
  inline void reset() {
    this->map_syms_ptrs.clear();
    this->map_syms_const_ptrs.clear();
  }

  /**
   *  Create instance from map from Syms to ParticleDats
   *
   *  @param sycl_target Compute device for all ParticleDats and buffers.
   *  @param particle_dats_map Map from Syms to ParticleDats.
   */
  SymVectorPointerCache(
      SYCLTargetSharedPtr sycl_target,
      std::map<Sym<T>, ParticleDatSharedPtr<T>> *particle_dats_map)
      : sycl_target(sycl_target), particle_dats_map(particle_dats_map) {
    this->reset();
  }

  /**
   * @param syms Sym vector to check existance for in cache.
   * @returns True if Sym vector exists in cache.
   */
  inline bool in_cache(std::vector<Sym<T>> &syms) {
    return this->map_syms_ptrs.count(syms);
  }
  /**
   * @param syms Sym vector to check existance for in cache.
   * @returns True if Sym vector exists in cache.
   */
  inline bool in_const_cache(std::vector<Sym<T>> &syms) {
    return this->map_syms_const_ptrs.count(syms);
  }

  /**
   * Get the device pointers that correspond to the ParticleDats requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetT<T> *get(std::vector<Sym<T>> &syms) {
    const std::size_t n = syms.size();
    if (!this->in_cache(syms)) {
      std::vector<ParticleDatImplGetT<T>> ptrs(n);
      for (std::size_t ix = 0; ix < n; ix++) {
        ptrs[ix] = this->particle_dats_map->at(syms[ix])->impl_get();
      }
      this->map_syms_ptrs[syms] =
          std::make_unique<BufferDevice<ParticleDatImplGetT<T>>>(
              this->sycl_target, ptrs);
    } else {
      // This ensures that impl_get is called on the dat incase that has
      // additional effects.
      for (std::size_t ix = 0; ix < n; ix++) {
        this->particle_dats_map->at(syms[ix])->impl_get();
      }
    }

    return this->map_syms_ptrs[syms]->ptr;
  }

  /**
   * Get the const device pointers that correspond to the ParticleDats
   * requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetConstT<T> *get_const(std::vector<Sym<T>> &syms) {

    const std::size_t n = syms.size();
    if (!this->in_const_cache(syms)) {
      std::vector<ParticleDatImplGetConstT<T>> ptrs(n);
      for (std::size_t ix = 0; ix < n; ix++) {
        ptrs[ix] = this->particle_dats_map->at(syms[ix])->impl_get_const();
      }
      this->map_syms_const_ptrs[syms] =
          std::make_unique<BufferDevice<ParticleDatImplGetConstT<T>>>(
              this->sycl_target, ptrs);
    } else {
      // This ensures that impl_get is called on the dat incase that has
      // additional effects.
      for (std::size_t ix = 0; ix < n; ix++) {
        this->particle_dats_map->at(syms[ix])->impl_get_const();
      }
    }

    return this->map_syms_const_ptrs[syms]->ptr;
  }
};

} // namespace NESO::Particles
#endif
