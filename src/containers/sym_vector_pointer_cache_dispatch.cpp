#include <neso_particles/containers/sym_vector_pointer_cache_dispatch.hpp>
#include <neso_particles/containers/sym_vector_pointer_cache_dispatch_impl.hpp>

namespace NESO::Particles {

void SymVectorPointerCacheDispatch::reset() {
  this->cache_int.reset();
  this->cache_real.reset();
}

void SymVectorPointerCacheDispatch::reset_ephemeral() {
  this->cache_int.reset_ephemeral();
  this->cache_real.reset_ephemeral();
}

SymVectorPointerCacheDispatch::SymVectorPointerCacheDispatch(
    SYCLTargetSharedPtr sycl_target,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_map_int,
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> *particle_dats_map_int_eph,
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_map_real,
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> *particle_dats_map_real_eph)
    : cache_int(sycl_target, particle_dats_map_int, particle_dats_map_int_eph),
      cache_real(sycl_target, particle_dats_map_real,
                 particle_dats_map_real_eph) {}

void SymVectorPointerCacheDispatch::create(std::vector<Sym<INT>> &syms) {
  this->cache_int.create(syms);
}

void SymVectorPointerCacheDispatch::create(std::vector<Sym<REAL>> &syms) {
  this->cache_real.create(syms);
}

void SymVectorPointerCacheDispatch::create_const(std::vector<Sym<INT>> &syms) {
  this->cache_int.create_const(syms);
}

void SymVectorPointerCacheDispatch::create_const(std::vector<Sym<REAL>> &syms) {
  this->cache_real.create_const(syms);
}

ParticleDatImplGetT<INT> *
SymVectorPointerCacheDispatch::get(std::vector<Sym<INT>> &syms) {
  return this->cache_int.get(syms);
}

ParticleDatImplGetT<REAL> *
SymVectorPointerCacheDispatch::get(std::vector<Sym<REAL>> &syms) {
  return this->cache_real.get(syms);
}

ParticleDatImplGetConstT<INT> *
SymVectorPointerCacheDispatch::get_const(std::vector<Sym<INT>> &syms) {
  return this->cache_int.get_const(syms);
}

ParticleDatImplGetConstT<REAL> *
SymVectorPointerCacheDispatch::get_const(std::vector<Sym<REAL>> &syms) {
  return this->cache_real.get_const(syms);
}

ParticleDatImplGetT<INT> SymVectorPointerCacheDispatch::get(Sym<INT> sym) {
  return this->cache_int.get(sym);
}

ParticleDatImplGetT<REAL> SymVectorPointerCacheDispatch::get(Sym<REAL> sym) {
  return this->cache_real.get(sym);
}

ParticleDatImplGetConstT<INT>
SymVectorPointerCacheDispatch::get_const(Sym<INT> sym) {
  return this->cache_int.get_const(sym);
}

ParticleDatImplGetConstT<REAL>
SymVectorPointerCacheDispatch::get_const(Sym<REAL> sym) {
  return this->cache_real.get_const(sym);
}

} // namespace NESO::Particles
