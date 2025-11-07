#ifndef _NESO_PARTICLES_PARTICLE_GROUP_POINTER_MAP_H__
#define _NESO_PARTICLES_PARTICLE_GROUP_POINTER_MAP_H__

#include "device_buffers.hpp"
#include "particle_dat.hpp"
#include "typedefs.hpp"
#include <map>

namespace NESO::Particles {

struct ParticleGroupPointerMapDeviceConst {
  REAL *const *const **d_ptr_real;
  INT *const *const **d_ptr_int;
  int *d_ncomp_real;
  int *d_ncomp_int;
  int *d_ncomp_exscan_real;
  int *d_ncomp_exscan_int;
  int *h_ncomp_real;
  int *h_ncomp_int;
  int *h_ncomp_exscan_real;
  int *h_ncomp_exscan_int;
  int ndat_real;
  int ndat_int;
  int ncomp_total_real;
  int ncomp_total_int;
  int *d_flattened_dat_index_real;
  int *d_flattened_dat_index_int;
  int *d_flattened_comp_index_real;
  int *d_flattened_comp_index_int;
  int *h_flattened_dat_index_real;
  int *h_flattened_dat_index_int;
  int *h_flattened_comp_index_real;
  int *h_flattened_comp_index_int;
};

struct ParticleGroupPointerMapDevice {
  REAL ****d_ptr_real;
  INT ****d_ptr_int;
  int *d_ncomp_real;
  int *d_ncomp_int;
  int *d_ncomp_exscan_real;
  int *d_ncomp_exscan_int;
  int *h_ncomp_real;
  int *h_ncomp_int;
  int *h_ncomp_exscan_real;
  int *h_ncomp_exscan_int;
  int ndat_real;
  int ndat_int;
  int ncomp_total_real;
  int ncomp_total_int;
  int *d_flattened_dat_index_real;
  int *d_flattened_dat_index_int;
  int *d_flattened_comp_index_real;
  int *d_flattened_comp_index_int;
  int *h_flattened_dat_index_real;
  int *h_flattened_dat_index_int;
  int *h_flattened_comp_index_real;
  int *h_flattened_comp_index_int;
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
  bool valid_const{false};
  bool valid{false};

  std::shared_ptr<BufferDeviceHost<REAL *const *const *>> dh_dat_ptr_const_real;
  std::shared_ptr<BufferDeviceHost<INT *const *const *>> dh_dat_ptr_const_int;
  std::shared_ptr<BufferDeviceHost<REAL ***>> dh_dat_ptr_real;
  std::shared_ptr<BufferDeviceHost<INT ***>> dh_dat_ptr_int;

  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_int;
  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_exscan_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_dat_ncomp_exscan_int;

  std::shared_ptr<BufferDeviceHost<int>> dh_flattened_dat_index_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_flattened_dat_index_int;
  std::shared_ptr<BufferDeviceHost<int>> dh_flattened_comp_index_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_flattened_comp_index_int;

  int ncomp_total_real;
  int ncomp_total_int;

  ParticleGroupPointerMapDeviceConst h_cache_const;
  ParticleGroupPointerMapDevice h_cache;

  std::shared_ptr<BufferDevice<ParticleGroupPointerMapDeviceConst>>
      d_cache_const;
  std::shared_ptr<BufferDevice<ParticleGroupPointerMapDevice>> d_cache;

  void create_const();
  void create();

public:
  /// Disable (implicit) copies.
  ParticleGroupPointerMap(const ParticleGroupPointerMap &st) = delete;
  /// Disable (implicit) copies.
  ParticleGroupPointerMap &operator=(ParticleGroupPointerMap const &a) = delete;

  std::vector<Sym<REAL>> map_index_to_sym_real;
  std::vector<Sym<INT>> map_index_to_sym_int;
  std::map<Sym<REAL>, int> map_sym_to_index_real;
  std::map<Sym<INT>, int> map_sym_to_index_int;

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
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_int);

  /**
   * Invalidate the cached values. Should be called when the maps are modified.
   */
  inline void invalidate() {
    this->valid_const = false;
    this->valid = false;
  }

  /**
   * @returns Device copyable type that describes read access to the dats.
   */
  ParticleGroupPointerMapDeviceConst &get_const();

  /**
   * @returns Device copyable type that describes write access to the dats.
   */
  ParticleGroupPointerMapDevice &get();

  /**
   * @returns Device pointer that describes read access to the dats.
   */
  ParticleGroupPointerMapDeviceConst *get_const_device();

  /**
   * @returns Device pointer that describes write access to the dats.
   */
  ParticleGroupPointerMapDevice *get_device();
};

} // namespace NESO::Particles

#endif
