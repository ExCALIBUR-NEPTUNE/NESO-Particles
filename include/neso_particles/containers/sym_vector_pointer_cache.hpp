#ifndef _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_H_
#define _NESO_PARTICLES_SYM_VECTOR_POINTER_CACHE_H_

#include "../compute_target.hpp"
#include "../particle_dat.hpp"
#include <map>
#include <memory>
#include <vector>
#include <set>

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
  /// Pointer to the ParticleGroup map from Sym to ParticleDat for ephemeral
  /// dats.
  std::map<Sym<T>, ParticleDatSharedPtr<T>> *particle_dats_map_eph;
  std::map<std::vector<Sym<T>>,
           std::unique_ptr<BufferDevice<ParticleDatImplGetT<T>>>>
      map_syms_ptrs;
  std::map<std::vector<Sym<T>>,
           std::unique_ptr<BufferDevice<ParticleDatImplGetConstT<T>>>>
      map_syms_const_ptrs;

  // Keys that contain EphemeralDats.
  std::set<std::vector<Sym<T>>> ephemeral_keys;

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
   * Empty the cache of any keys which contained EphemeralDat Syms.
   */
  inline void reset_ephemeral() {
    for (auto &keyx : this->ephemeral_keys) {
      this->map_syms_ptrs.erase(keyx);
      this->map_syms_const_ptrs.erase(keyx);
    }
    this->ephemeral_keys.clear();
  }

  /**
   *  Create instance from map from Syms to ParticleDats
   *
   *  @param sycl_target Compute device for all ParticleDats and buffers.
   *  @param particle_dats_map Map from Syms to ParticleDats.
   *  @param particle_dats_map_eph Map from Syms to ParticleDats for
   * EphemeralDats.
   */
  SymVectorPointerCache(
      SYCLTargetSharedPtr sycl_target,
      std::map<Sym<T>, ParticleDatSharedPtr<T>> *particle_dats_map,
      std::map<Sym<T>, ParticleDatSharedPtr<T>> *particle_dats_map_eph)
      : sycl_target(sycl_target), particle_dats_map(particle_dats_map),
        particle_dats_map_eph(particle_dats_map_eph) {
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
   * Find the ParticleDat for a given Sym. First checks the second map for
   * EphermalDats, if the map is not a nullptr, then checks the first.
   *
   * @param[in] sym Sym to find corresponding ParticleDat for.
   * @param[in, out] is_ephemeral Optional bool, if not nullptr then this bool
   * is true on return if the returned dat is an EphermalDat.
   * @returns ParticleDat.
   */
  inline ParticleDatSharedPtr<T> find(Sym<T> sym,
                                      bool *is_ephemeral = nullptr) {
    ParticleDatSharedPtr<T> dat = nullptr;
    bool is_ephemeral_t = false;

    if (this->particle_dats_map_eph != nullptr) {
      if (this->particle_dats_map_eph->count(sym)) {
        dat = this->particle_dats_map_eph->at(sym);
        is_ephemeral_t = true;
      }
    }

    if (dat == nullptr) {
      if (this->particle_dats_map->count(sym)) {
        dat = this->particle_dats_map->at(sym);
        is_ephemeral_t = false;
      }
    }

    if (is_ephemeral != nullptr) {
      *is_ephemeral = is_ephemeral_t;
    }

    NESOASSERT(
        dat != nullptr,
        "Could not find ParticleDat or EphemeralDat for Sym with name: " +
            sym.name);

    return dat;
  }

  /**
   * Get the ParticleDat device type that corresponds to a Sym<T> for read-write
   * access. This method first checks the EphemeralDats for the requested sym
   * then the base ParticleGroup dats.
   *
   * @param sym Sym<T> to find corresponding dat for.
   * @returns ParticleDat device type for requested Sym.
   */
  inline ParticleDatImplGetT<T> get(Sym<T> sym) {
    return this->find(sym)->impl_get();
  }

  /**
   * Get the ParticleDat device type that corresponds to a Sym<T> for read only
   * access. This method first checks the EphemeralDats for the requested sym
   * then the base ParticleGroup dats.
   *
   * @param sym Sym<T> to find corresponding dat for.
   * @returns ParticleDat device type for requested Sym.
   */
  inline ParticleDatImplGetConstT<T> get_const(Sym<T> sym) {
    return this->find(sym)->impl_get_const();
  }

  /**
   * Create the cache entry for a vector of Syms.
   *
   * @param syms Syms to create entry for.
   */
  inline void create(std::vector<Sym<T>> &syms) {
    if (!this->in_cache(syms)) {
      const std::size_t n = syms.size();
      std::vector<ParticleDatImplGetT<T>> ptrs(n);
      bool is_ephemeral = false;
      for (std::size_t ix = 0; ix < n; ix++) {
        // Use d_ptr to avoid side effects from impl_get
        bool is_ephemeral_inner;
        ptrs[ix] = this->find(syms[ix], &is_ephemeral_inner)->cell_dat.d_ptr;
        is_ephemeral = is_ephemeral || is_ephemeral_inner;
      }
      this->map_syms_ptrs[syms] =
          std::make_unique<BufferDevice<ParticleDatImplGetT<T>>>(
              this->sycl_target, ptrs);
      if (is_ephemeral) {
        this->ephemeral_keys.insert(syms);
      }
    }
  }

  /**
   * Create the cache entry for a vector of Syms for const access.
   *
   * @param syms Syms to create entry for.
   */
  inline void create_const(std::vector<Sym<T>> &syms) {
    if (!this->in_const_cache(syms)) {
      const std::size_t n = syms.size();
      std::vector<ParticleDatImplGetConstT<T>> ptrs(n);

      bool is_ephemeral = false;
      for (std::size_t ix = 0; ix < n; ix++) {
        // Use d_ptr to avoid side effects from impl_get
        bool is_ephemeral_inner;
        ptrs[ix] = this->find(syms[ix], &is_ephemeral_inner)->cell_dat.d_ptr;
        is_ephemeral = is_ephemeral || is_ephemeral_inner;
      }
      this->map_syms_const_ptrs[syms] =
          std::make_unique<BufferDevice<ParticleDatImplGetConstT<T>>>(
              this->sycl_target, ptrs);
      if (is_ephemeral) {
        this->ephemeral_keys.insert(syms);
      }
    }
  }

  /**
   * Get the device pointers that correspond to the ParticleDats requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetT<T> *get(std::vector<Sym<T>> &syms) {
    // This ensures that impl_get is called on the dat incase that has
    // additional effects.
    const std::size_t n = syms.size();
    for (std::size_t ix = 0; ix < n; ix++) {
      this->get(syms[ix]);
    }

    return this->map_syms_ptrs.at(syms)->ptr;
  }

  /**
   * Get the const device pointers that correspond to the ParticleDats
   * requested.
   *
   * @param syms Syms to get ParticleDats for.
   */
  inline ParticleDatImplGetConstT<T> *get_const(std::vector<Sym<T>> &syms) {
    // This ensures that impl_get is called on the dat incase that has
    // additional effects.
    const std::size_t n = syms.size();
    for (std::size_t ix = 0; ix < n; ix++) {
      this->get_const(syms[ix]);
    }

    return this->map_syms_const_ptrs.at(syms)->ptr;
  }
};

} // namespace NESO::Particles
#endif
