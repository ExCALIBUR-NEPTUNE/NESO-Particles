#include <neso_particles/particle_group_pointer_map.hpp>

namespace NESO::Particles {

void ParticleGroupPointerMap::create_const() {
  if (!this->valid_const) {
    this->map_index_to_sym_real.clear();
    this->map_index_to_sym_int.clear();
    this->map_sym_to_index_real.clear();
    this->map_sym_to_index_int.clear();

    const std::size_t ndat_real = this->particle_dats_real->size();
    const std::size_t ndat_int = this->particle_dats_int->size();
    this->dh_dat_ptr_const_real->realloc_no_copy(ndat_real);
    this->dh_dat_ptr_const_int->realloc_no_copy(ndat_int);
    this->dh_dat_ncomp_real->realloc_no_copy(ndat_real);
    this->dh_dat_ncomp_int->realloc_no_copy(ndat_int);
    this->dh_dat_ncomp_exscan_real->realloc_no_copy(ndat_real);
    this->dh_dat_ncomp_exscan_int->realloc_no_copy(ndat_int);

    this->ncomp_total_real = 0;
    int index = 0;
    for (auto [sym, dat] : *this->particle_dats_real) {
      const int ncomp = dat->ncomp;
      this->dh_dat_ptr_const_real->h_buffer.ptr[index] = dat->impl_get_const();
      this->dh_dat_ncomp_real->h_buffer.ptr[index] = ncomp;
      this->dh_dat_ncomp_exscan_real->h_buffer.ptr[index] =
          this->ncomp_total_real;
      this->ncomp_total_real += ncomp;
      this->map_index_to_sym_real.push_back(sym);
      this->map_sym_to_index_real[sym] = index;
      index++;
    }

    this->dh_flattened_dat_index_real->realloc_no_copy(this->ncomp_total_real);
    this->dh_flattened_comp_index_real->realloc_no_copy(this->ncomp_total_real);

    index = 0;
    int dat_index = 0;
    for (auto [sym, dat] : *this->particle_dats_real) {
      const int ncomp = dat->ncomp;
      for (int cx = 0; cx < ncomp; cx++) {
        this->dh_flattened_dat_index_real->h_buffer.ptr[index] = dat_index;
        this->dh_flattened_comp_index_real->h_buffer.ptr[index] = cx;
        index++;
      }
      dat_index++;
    }

    this->ncomp_total_int = 0;
    index = 0;
    for (auto [sym, dat] : *this->particle_dats_int) {
      const int ncomp = dat->ncomp;
      this->dh_dat_ptr_const_int->h_buffer.ptr[index] = dat->impl_get_const();
      this->dh_dat_ncomp_int->h_buffer.ptr[index] = ncomp;
      this->dh_dat_ncomp_exscan_int->h_buffer.ptr[index] =
          this->ncomp_total_int;
      this->ncomp_total_int += ncomp;
      this->map_index_to_sym_int.push_back(sym);
      this->map_sym_to_index_int[sym] = index;
      index++;
    }

    this->dh_flattened_dat_index_int->realloc_no_copy(this->ncomp_total_int);
    this->dh_flattened_comp_index_int->realloc_no_copy(this->ncomp_total_int);

    index = 0;
    dat_index = 0;
    for (auto [sym, dat] : *this->particle_dats_int) {
      const int ncomp = dat->ncomp;
      for (int cx = 0; cx < ncomp; cx++) {
        this->dh_flattened_dat_index_int->h_buffer.ptr[index] = dat_index;
        this->dh_flattened_comp_index_int->h_buffer.ptr[index] = cx;
        index++;
      }
      dat_index++;
    }

    this->dh_dat_ptr_const_real->host_to_device();
    this->dh_dat_ptr_const_int->host_to_device();
    this->dh_dat_ncomp_real->host_to_device();
    this->dh_dat_ncomp_int->host_to_device();
    this->dh_dat_ncomp_exscan_real->host_to_device();
    this->dh_dat_ncomp_exscan_int->host_to_device();
    this->dh_flattened_dat_index_real->host_to_device();
    this->dh_flattened_comp_index_real->host_to_device();
    this->dh_flattened_dat_index_int->host_to_device();
    this->dh_flattened_comp_index_int->host_to_device();

    // Creat the device copyable struct
    this->h_cache_const.d_ptr_real = this->dh_dat_ptr_const_real->d_buffer.ptr;
    this->h_cache_const.d_ptr_int = this->dh_dat_ptr_const_int->d_buffer.ptr;

    this->h_cache_const.d_ncomp_real = this->dh_dat_ncomp_real->d_buffer.ptr;
    this->h_cache_const.d_ncomp_int = this->dh_dat_ncomp_int->d_buffer.ptr;
    this->h_cache_const.d_ncomp_exscan_real =
        this->dh_dat_ncomp_exscan_real->d_buffer.ptr;
    this->h_cache_const.d_ncomp_exscan_int =
        this->dh_dat_ncomp_exscan_int->d_buffer.ptr;

    this->h_cache_const.h_ncomp_real = this->dh_dat_ncomp_real->h_buffer.ptr;
    this->h_cache_const.h_ncomp_int = this->dh_dat_ncomp_int->h_buffer.ptr;
    this->h_cache_const.h_ncomp_exscan_real =
        this->dh_dat_ncomp_exscan_real->h_buffer.ptr;
    this->h_cache_const.h_ncomp_exscan_int =
        this->dh_dat_ncomp_exscan_int->h_buffer.ptr;

    this->h_cache_const.ndat_real = static_cast<int>(ndat_real);
    this->h_cache_const.ndat_int = static_cast<int>(ndat_int);
    this->h_cache_const.ncomp_total_real = this->ncomp_total_real;
    this->h_cache_const.ncomp_total_int = this->ncomp_total_int;

    this->h_cache_const.d_flattened_dat_index_real =
        this->dh_flattened_dat_index_real->d_buffer.ptr;
    this->h_cache_const.d_flattened_dat_index_int =
        this->dh_flattened_dat_index_int->d_buffer.ptr;
    this->h_cache_const.d_flattened_comp_index_real =
        this->dh_flattened_comp_index_real->d_buffer.ptr;
    this->h_cache_const.d_flattened_comp_index_int =
        this->dh_flattened_comp_index_int->d_buffer.ptr;
    this->h_cache_const.h_flattened_dat_index_real =
        this->dh_flattened_dat_index_real->h_buffer.ptr;
    this->h_cache_const.h_flattened_dat_index_int =
        this->dh_flattened_dat_index_int->h_buffer.ptr;
    this->h_cache_const.h_flattened_comp_index_real =
        this->dh_flattened_comp_index_real->h_buffer.ptr;
    this->h_cache_const.h_flattened_comp_index_int =
        this->dh_flattened_comp_index_int->h_buffer.ptr;

    this->d_cache_const->set(std::vector{this->h_cache_const});
    this->valid_const = true;
  }
}

void ParticleGroupPointerMap::create() {
  this->create_const();
  if (!this->valid) {

    const std::size_t ndat_real = this->particle_dats_real->size();
    const std::size_t ndat_int = this->particle_dats_int->size();

    this->dh_dat_ptr_real->realloc_no_copy(ndat_real);
    this->dh_dat_ptr_int->realloc_no_copy(ndat_int);

    int index = 0;
    for (auto [sym, dat] : *this->particle_dats_real) {
      this->dh_dat_ptr_real->h_buffer.ptr[index] = dat->impl_get();
      index++;
    }
    index = 0;
    for (auto [sym, dat] : *this->particle_dats_int) {
      this->dh_dat_ptr_int->h_buffer.ptr[index] = dat->impl_get();
      index++;
    }

    this->dh_dat_ptr_real->host_to_device();
    this->dh_dat_ptr_int->host_to_device();

    this->h_cache.d_ptr_real = this->dh_dat_ptr_real->d_buffer.ptr;
    this->h_cache.d_ptr_int = this->dh_dat_ptr_int->d_buffer.ptr;
    this->h_cache.d_ncomp_real = this->h_cache_const.d_ncomp_real;
    this->h_cache.d_ncomp_int = this->h_cache_const.d_ncomp_int;
    this->h_cache.d_ncomp_exscan_real = this->h_cache_const.d_ncomp_exscan_real;
    this->h_cache.d_ncomp_exscan_int = this->h_cache_const.d_ncomp_exscan_int;
    this->h_cache.h_ncomp_real = this->h_cache_const.h_ncomp_real;
    this->h_cache.h_ncomp_int = this->h_cache_const.h_ncomp_int;
    this->h_cache.h_ncomp_exscan_real = this->h_cache_const.h_ncomp_exscan_real;
    this->h_cache.h_ncomp_exscan_int = this->h_cache_const.h_ncomp_exscan_int;
    this->h_cache.ndat_real = this->h_cache_const.ndat_real;
    this->h_cache.ndat_int = this->h_cache_const.ndat_int;
    this->h_cache.ncomp_total_real = this->h_cache_const.ncomp_total_real;
    this->h_cache.ncomp_total_int = this->h_cache_const.ncomp_total_int;

    this->h_cache.d_flattened_dat_index_real =
        this->h_cache_const.d_flattened_dat_index_real;
    this->h_cache.d_flattened_dat_index_int =
        this->h_cache_const.d_flattened_dat_index_int;
    this->h_cache.d_flattened_comp_index_real =
        this->h_cache_const.d_flattened_comp_index_real;
    this->h_cache.d_flattened_comp_index_int =
        this->h_cache_const.d_flattened_comp_index_int;
    this->h_cache.h_flattened_dat_index_real =
        this->h_cache_const.h_flattened_dat_index_real;
    this->h_cache.h_flattened_dat_index_int =
        this->h_cache_const.h_flattened_dat_index_int;
    this->h_cache.h_flattened_comp_index_real =
        this->h_cache_const.h_flattened_comp_index_real;
    this->h_cache.h_flattened_comp_index_int =
        this->h_cache_const.h_flattened_comp_index_int;

    this->d_cache->set(std::vector{this->h_cache});

    this->valid = true;
  } else {
    for (auto [sym, dat] : *this->particle_dats_real) {
      dat->impl_get();
    }
    for (auto [sym, dat] : *this->particle_dats_int) {
      dat->impl_get();
    }
  }
}

ParticleGroupPointerMap::ParticleGroupPointerMap(
    SYCLTargetSharedPtr sycl_target,
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_int)
    : sycl_target(sycl_target), particle_dats_real(particle_dats_real),
      particle_dats_int(particle_dats_int), valid_const(false) {

  this->dh_dat_ptr_const_real =
      std::make_shared<BufferDeviceHost<REAL *const *const *>>(
          this->sycl_target, 1);
  this->dh_dat_ptr_const_int =
      std::make_shared<BufferDeviceHost<INT *const *const *>>(this->sycl_target,
                                                              1);

  this->dh_dat_ptr_real =
      std::make_shared<BufferDeviceHost<REAL ***>>(this->sycl_target, 1);
  this->dh_dat_ptr_int =
      std::make_shared<BufferDeviceHost<INT ***>>(this->sycl_target, 1);

  this->dh_dat_ncomp_real =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);
  this->dh_dat_ncomp_int =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);
  this->dh_dat_ncomp_exscan_real =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);
  this->dh_dat_ncomp_exscan_int =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);

  this->dh_flattened_dat_index_real =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);
  this->dh_flattened_dat_index_int =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);
  this->dh_flattened_comp_index_real =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);
  this->dh_flattened_comp_index_int =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, 1);

  this->d_cache_const =
      std::make_shared<BufferDevice<ParticleGroupPointerMapDeviceConst>>(
          this->sycl_target, std::vector{this->h_cache_const});
  this->d_cache = std::make_shared<BufferDevice<ParticleGroupPointerMapDevice>>(
      this->sycl_target, std::vector{this->h_cache});
}

ParticleGroupPointerMapDeviceConst &ParticleGroupPointerMap::get_const() {
  this->create_const();
  return this->h_cache_const;
}

ParticleGroupPointerMapDevice &ParticleGroupPointerMap::get() {
  this->create();
  return this->h_cache;
}

void ParticleGroupPointerMap::get_const_device(
    ParticleGroupPointerMapDeviceConst **h_map,
    ParticleGroupPointerMapDeviceConst **d_map) {
  this->create_const();
  *h_map = &this->h_cache_const;
  *d_map = this->d_cache_const->ptr;
}

void ParticleGroupPointerMap::get_device(
    ParticleGroupPointerMapDevice **h_map,
    ParticleGroupPointerMapDevice **d_map) {
  this->create();
  *h_map = &this->h_cache;
  *d_map = this->d_cache->ptr;
}

} // namespace NESO::Particles
