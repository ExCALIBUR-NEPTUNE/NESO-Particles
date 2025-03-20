#ifndef _NESO_PARTICLES_PARTICLE_GROUP_POINTER_MAP_H__
#define _NESO_PARTICLES_PARTICLE_GROUP_POINTER_MAP_H__

#include "device_buffers.hpp"
#include "particle_dat.hpp"
#include "typedefs.hpp"
#include <map>

namespace NESO::Particles {

struct ParticleGroupPointerMapDevice {
  REAL *const *const **d_ptr_real;
  INT *const *const **d_ptr_int;
  int *d_ncomp_real;
  int *d_ncomp_int;
  int *d_ncomp_exscan_real;
  int *d_ncomp_exscan_int;
  int ndat_real;
  int ndat_int;
  int ncomp_total_real;
  int ncomp_total_int;
};

/**
 * Type to create and cache the device information required for low level dat
 * access.
 */
class ParticleGroupPointerMap {
protected:
  SYCLTargetSharedPtr sycl_target;
  std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_real;
  std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_int;
  bool valid;

  std::shared_ptr<BufferDeviceHost<REAL *const *const *>> dh_dat_ptr_real;
  std::shared_ptr<BufferDeviceHost<INT *const *const *>> dh_dat_ptr_int;
  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_int;
  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_exscan_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_exscan_int;

  int ncomp_total_real;
  int ncomp_total_int;

  ParticleGroupPointerMapDevice d_cache;

public:
  /// Disable (implicit) copies.
  ParticleGroupPointerMap(const ParticleGroupPointerMap &st) = delete;
  /// Disable (implicit) copies.
  ParticleGroupPointerMap &operator=(ParticleGroupPointerMap const &a) = delete;

  /**
   * Create new instance.
   *
   * @param sycl_target SYCLTarget for device.
   * @param particle_dats_real Pointer to ParticleGroup REAL dats.
   * @param particle_dats_int Pointer to ParticleGroup INT dats.
   */
  ParticleGroupPointerMap(
      SYCLTargetSharedPtr sycl_target,
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_real,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_int)
      : sycl_target(sycl_target), particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int), valid(false) {

    const std::size_t ndat_real =
        std::max(this->particle_dats_real->size(), (std::size_t)1);
    const std::size_t ndat_int =
        std::max(this->particle_dats_int->size(), (std::size_t)2);

    this->dh_dat_ptr_real =
        std::make_shared<BufferDeviceHost<REAL *const *const *>>(
            this->sycl_target, ndat_real);
    this->dh_dat_ptr_int =
        std::make_shared<BufferDeviceHost<INT *const *const *>>(
            this->sycl_target, ndat_int);
    this->dh_dat_ncomp_real =
        std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_real);
    this->dh_dat_ncomp_int =
        std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_int);
    this->dh_dat_ncomp_exscan_real =
        std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_real);
    this->dh_dat_ncomp_exscan_int =
        std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_int);
  }

  /**
   * Invalidate the cached values. Should be called when the maps are modified.
   */
  inline void invalidate() { this->valid = false; }

  /**
   * @returns Device copyable type that describes access to the dats.
   */
  inline ParticleGroupPointerMapDevice get() {
    if (!this->valid) {
      const std::size_t ndat_real = this->particle_dats_real->size();
      const std::size_t ndat_int = this->particle_dats_int->size();
      this->dh_dat_ptr_real->realloc_no_copy(ndat_real);
      this->dh_dat_ptr_int->realloc_no_copy(ndat_int);
      this->dh_dat_ncomp_real->realloc_no_copy(ndat_real);
      this->dh_dat_ncomp_int->realloc_no_copy(ndat_int);
      this->dh_dat_ncomp_exscan_real->realloc_no_copy(ndat_real);
      this->dh_dat_ncomp_exscan_int->realloc_no_copy(ndat_int);

      this->ncomp_total_real = 0;
      int index = 0;
      for (auto [sym, dat] : *this->particle_dats_real) {
        const int ncomp = dat->ncomp;
        this->dh_dat_ptr_real->h_buffer.ptr[index] = dat->impl_get_const();
        this->dh_dat_ncomp_real->h_buffer.ptr[index] = ncomp;
        this->dh_dat_ncomp_exscan_real->h_buffer.ptr[index] =
            this->ncomp_total_real;
        this->ncomp_total_real += ncomp;
        index++;
      }
      this->ncomp_total_int = 0;
      index = 0;
      for (auto [sym, dat] : *this->particle_dats_int) {
        const int ncomp = dat->ncomp;
        this->dh_dat_ptr_int->h_buffer.ptr[index] = dat->impl_get_const();
        this->dh_dat_ncomp_int->h_buffer.ptr[index] = ncomp;
        this->dh_dat_ncomp_exscan_int->h_buffer.ptr[index] =
            this->ncomp_total_int;
        this->ncomp_total_int += ncomp;
        index++;
      }

      this->dh_dat_ptr_real->host_to_device();
      this->dh_dat_ptr_int->host_to_device();
      this->dh_dat_ncomp_real->host_to_device();
      this->dh_dat_ncomp_int->host_to_device();
      this->dh_dat_ncomp_exscan_real->host_to_device();
      this->dh_dat_ncomp_exscan_int->host_to_device();
      // Creat the device copyable struct
      this->d_cache.d_ptr_real = this->dh_dat_ptr_real->d_buffer.ptr;
      this->d_cache.d_ptr_int = this->dh_dat_ptr_int->d_buffer.ptr;
      this->d_cache.d_ncomp_real = this->dh_dat_ncomp_real->d_buffer.ptr;
      this->d_cache.d_ncomp_int = this->dh_dat_ncomp_int->d_buffer.ptr;
      this->d_cache.d_ncomp_exscan_real =
          this->dh_dat_ncomp_exscan_real->d_buffer.ptr;
      this->d_cache.d_ncomp_exscan_int =
          this->dh_dat_ncomp_exscan_int->d_buffer.ptr;
      this->d_cache.ndat_real = static_cast<int>(ndat_real);
      this->d_cache.ndat_int = static_cast<int>(ndat_int);
      this->d_cache.ncomp_total_real = this->ncomp_total_real;
      this->d_cache.ncomp_total_int = this->ncomp_total_int;
      this->valid = true;
    }
    return this->d_cache;
  }
};

} // namespace NESO::Particles

#endif
