#include <neso_particles/containers/ephemeral_dats.hpp>

namespace NESO::Particles {
void EphemeralDats::reset_ephemeral_dats(INT npart_local, int *h_npart_cell,
                                         int *d_npart_cell,
                                         INT *d_npart_cell_es) {
  this->ephemeral.es.wait();
  this->ephemeral.npart_local = npart_local;
  this->ephemeral.h_npart_cell = h_npart_cell;
  this->ephemeral.d_npart_cell = d_npart_cell;
  this->ephemeral.d_npart_cell_es = d_npart_cell_es;
  this->ephemeral.dats_real.clear();
  this->ephemeral.dats_int.clear();
  this->ephemeral.sym_vector_pointer_cache_dispatch->reset_ephemeral();
}

void EphemeralDats::push_ephemeral_dat(Sym<REAL> sym,
                                       ParticleDatSharedPtr<REAL> dat) {
  this->ephemeral.dats_real[sym] = dat;
}

void EphemeralDats::push_ephemeral_dat(Sym<INT> sym,
                                       ParticleDatSharedPtr<INT> dat) {
  this->ephemeral.dats_int[sym] = dat;
}

void EphemeralDats::pop_ephemeral_dat(Sym<REAL> sym) {
  this->ephemeral.dats_real.erase(sym);
}
void EphemeralDats::pop_ephemeral_dat(Sym<INT> sym) {
  this->ephemeral.dats_int.erase(sym);
}

template void EphemeralDats::add_ephemeral_dat(const Sym<REAL> sym,
                                               const int ncomp);
template void EphemeralDats::add_ephemeral_dat(const Sym<INT> sym,
                                               const int ncomp);
template void EphemeralDats::remove_ephemeral_dat(const Sym<REAL> sym);
template void EphemeralDats::remove_ephemeral_dat(const Sym<INT> sym);

bool EphemeralDats::contains_ephemeral_dat(Sym<REAL> sym) {
  this->invalidate_ephemeral_dats_if_required();
  return (bool)this->ephemeral.dats_real.count(sym);
}

bool EphemeralDats::contains_ephemeral_dat(Sym<INT> sym) {
  this->invalidate_ephemeral_dats_if_required();
  return (bool)this->ephemeral.dats_int.count(sym);
}

ParticleDatSharedPtr<REAL> EphemeralDats::get_ephemeral_dat(Sym<REAL> sym) {
  this->invalidate_ephemeral_dats_if_required();
  NESOASSERT(this->contains_ephemeral_dat(sym),
             "Cannot find EphemeralDat with name: " + sym.name);
  return this->ephemeral.dats_real.at(sym);
}

ParticleDatSharedPtr<INT> EphemeralDats::get_ephemeral_dat(Sym<INT> sym) {
  this->invalidate_ephemeral_dats_if_required();
  NESOASSERT(this->contains_ephemeral_dat(sym),
             "Cannot find EphemeralDat with name: " + sym.name);
  return this->ephemeral.dats_int.at(sym);
}

} // namespace NESO::Particles
