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
public:
  SYCLTargetSharedPtr sycl_target;
  std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_real;
  std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_int;

protected:
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

  // members for npart cell setting
  std::size_t cell_count{0};
  std::shared_ptr<BufferDevice<int *>> d_npart_cell_ptrs;
  std::vector<int *> h_npart_cell_ptrs;
  std::size_t num_bytes_per_particle{0};

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
   * @param[in, out] h_map Host pointer that describes read access to the dats.
   * @param[in, out] d_map Device pointer that describes read access to the
   * dats.
   */
  void get_const_device(ParticleGroupPointerMapDeviceConst **h_map,
                        ParticleGroupPointerMapDeviceConst **d_map);

  /**
   * @param[in, out] h_map Host pointer that describes write access to the dats.
   * @param[in, out] d_map Device pointer that describes write access to the
   * dats.
   */
  void get_device(ParticleGroupPointerMapDevice **h_map,
                  ParticleGroupPointerMapDevice **d_map);

  /**
   * Call to set h_npart_cells and d_npart_cells from a host and device buffer
   * for all particle dats.
   *
   * @param h_npart_cell_in Host accessible pointer to an array containing new
   * cell counts.
   * @param d_npart_cell_in Device accessible pointer to an array containing new
   * cell_counts.
   */
  template <typename U>
  inline void set_npart_cells_host_device(const U *h_npart_cell,
                                          const U *d_npart_cell) {
    // If you edit how this works check that
    // ParticleGroupPointerMap::set_npart_cells_device is still good.
    this->create();

    const std::size_t num_dats = this->h_npart_cell_ptrs.size();
    int **k_npart_cell_ptrs = this->d_npart_cell_ptrs->ptr;

    auto e0 = this->sycl_target->queue.parallel_for(
        this->sycl_target->device_limits.validate_range_global(
            sycl::range<2>(num_dats, this->cell_count)),
        [=](sycl::item<2> idx) {
          const std::size_t dat = idx.get_id(0);
          const std::size_t cell = idx.get_id(1);
          const int npart_cell = d_npart_cell[cell];
          k_npart_cell_ptrs[dat][cell] = npart_cell;
        });

    for (auto [sym, dat] : *this->particle_dats_real) {
      dat->write_callback_wrapper(0);
    }
    for (auto [sym, dat] : *this->particle_dats_int) {
      dat->write_callback_wrapper(0);
    }

    for (std::size_t dx = 0; dx < num_dats; dx++) {
      for (std::size_t cx = 0; cx < this->cell_count; cx++) {
        this->h_npart_cell_ptrs[dx][cx] = h_npart_cell[cx];
      }
    }

    e0.wait_and_throw();
  }

  /**
   * Call to set h_npart_cells and d_npart_cells from a host buffer for all
   * particle dats.
   *
   * @param h_npart_cell_in Host accessible pointer to an array containing new
   * cell counts.
   */
  template <typename U>
  inline void set_npart_cells_host(const U *h_npart_cell) {
    // This call ensures cell_count is set.
    this->create_const();
    auto d_npart_cell =
        get_resource<BufferDevice<U>, ResourceStackInterfaceBufferDevice<U>>(
            sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<U>{},
            sycl_target);
    d_npart_cell->realloc_no_copy(this->cell_count);
    this->sycl_target->queue
        .memcpy(d_npart_cell->ptr, h_npart_cell, this->cell_count * sizeof(U))
        .wait_and_throw();

    this->set_npart_cells_host_device(h_npart_cell, d_npart_cell->ptr);

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<U>{}, d_npart_cell);
  }

  /**
   * @returns The number of bytes per particle.
   */
  std::size_t get_num_bytes_per_particle();

  /**
   * @returns The cell count.
   */
  int get_cell_count();
};

using ParticleGroupPointerMapSharedPtr =
    std::shared_ptr<ParticleGroupPointerMap>;

} // namespace NESO::Particles

#endif
