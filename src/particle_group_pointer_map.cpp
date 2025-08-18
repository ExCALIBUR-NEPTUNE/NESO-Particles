#include <neso_particles/particle_group_pointer_map.hpp>

namespace NESO::Particles {

ParticleGroupPointerMap::ParticleGroupPointerMap(
    SYCLTargetSharedPtr sycl_target,
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_real,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_int)
    : sycl_target(sycl_target), particle_dats_real(particle_dats_real),
      particle_dats_int(particle_dats_int), valid(false) {

  const std::size_t ndat_real =
      std::max(this->particle_dats_real->size(), (std::size_t)1);
  const std::size_t ndat_int =
      std::max(this->particle_dats_int->size(), (std::size_t)2);

  this->dh_dat_ptr_const_real =
      std::make_shared<BufferDeviceHost<REAL *const *const *>>(
          this->sycl_target, ndat_real);
  this->dh_dat_ptr_const_int =
      std::make_shared<BufferDeviceHost<INT *const *const *>>(this->sycl_target,
                                                              ndat_int);

  this->dh_dat_ptr_real = std::make_shared<BufferDeviceHost<REAL ***>>(
      this->sycl_target, ndat_real);
  this->dh_dat_ptr_int =
      std::make_shared<BufferDeviceHost<INT ***>>(this->sycl_target, ndat_int);

  this->dh_dat_ncomp_real =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_real);
  this->dh_dat_ncomp_int =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_int);
  this->dh_dat_ncomp_exscan_real =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_real);
  this->dh_dat_ncomp_exscan_int =
      std::make_shared<BufferDeviceHost<int>>(this->sycl_target, ndat_int);
}

ParticleGroupPointerMapDeviceConst ParticleGroupPointerMap::get_const() {
  if (!this->valid) {
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

    this->dh_dat_ptr_const_real->host_to_device();
    this->dh_dat_ptr_const_int->host_to_device();
    this->dh_dat_ncomp_real->host_to_device();
    this->dh_dat_ncomp_int->host_to_device();
    this->dh_dat_ncomp_exscan_real->host_to_device();
    this->dh_dat_ncomp_exscan_int->host_to_device();
    // Creat the device copyable struct
    this->d_cache_const.d_ptr_real = this->dh_dat_ptr_const_real->d_buffer.ptr;
    this->d_cache_const.d_ptr_int = this->dh_dat_ptr_const_int->d_buffer.ptr;

    this->d_cache_const.d_ncomp_real = this->dh_dat_ncomp_real->d_buffer.ptr;
    this->d_cache_const.d_ncomp_int = this->dh_dat_ncomp_int->d_buffer.ptr;
    this->d_cache_const.d_ncomp_exscan_real =
        this->dh_dat_ncomp_exscan_real->d_buffer.ptr;
    this->d_cache_const.d_ncomp_exscan_int =
        this->dh_dat_ncomp_exscan_int->d_buffer.ptr;

    this->d_cache_const.h_ncomp_real = this->dh_dat_ncomp_real->h_buffer.ptr;
    this->d_cache_const.h_ncomp_int = this->dh_dat_ncomp_int->h_buffer.ptr;
    this->d_cache_const.h_ncomp_exscan_real =
        this->dh_dat_ncomp_exscan_real->h_buffer.ptr;
    this->d_cache_const.h_ncomp_exscan_int =
        this->dh_dat_ncomp_exscan_int->h_buffer.ptr;

    this->d_cache_const.ndat_real = static_cast<int>(ndat_real);
    this->d_cache_const.ndat_int = static_cast<int>(ndat_int);
    this->d_cache_const.ncomp_total_real = this->ncomp_total_real;
    this->d_cache_const.ncomp_total_int = this->ncomp_total_int;
    this->valid = true;
  }
  return this->d_cache_const;
}

ParticleGroupPointerMapDevice ParticleGroupPointerMap::get() {
  this->get_const();
  const auto ndat_real = this->d_cache_const.ndat_real;
  const auto ndat_int = this->d_cache_const.ndat_int;
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

  this->d_cache.d_ptr_real = this->dh_dat_ptr_real->d_buffer.ptr;
  this->d_cache.d_ptr_int = this->dh_dat_ptr_int->d_buffer.ptr;
  this->d_cache.d_ncomp_real = this->d_cache_const.d_ncomp_real;
  this->d_cache.d_ncomp_int = this->d_cache_const.d_ncomp_int;
  this->d_cache.d_ncomp_exscan_real = this->d_cache_const.d_ncomp_exscan_real;
  this->d_cache.d_ncomp_exscan_int = this->d_cache_const.d_ncomp_exscan_int;
  this->d_cache.h_ncomp_real = this->d_cache_const.h_ncomp_real;
  this->d_cache.h_ncomp_int = this->d_cache_const.h_ncomp_int;
  this->d_cache.h_ncomp_exscan_real = this->d_cache_const.h_ncomp_exscan_real;
  this->d_cache.h_ncomp_exscan_int = this->d_cache_const.h_ncomp_exscan_int;
  this->d_cache.ndat_real = this->d_cache_const.ndat_real;
  this->d_cache.ndat_int = this->d_cache_const.ndat_int;
  this->d_cache.ncomp_total_real = this->d_cache_const.ncomp_total_real;
  this->d_cache.ncomp_total_int = this->d_cache_const.ncomp_total_int;
  return this->d_cache;
}

} // namespace NESO::Particles
