#ifndef __NESO_PARTICLES_EPHEMERAL_EPHEMERAL_DATS_HPP_
#define __NESO_PARTICLES_EPHEMERAL_EPHEMERAL_DATS_HPP_

#include "../particle_dat.hpp"

namespace NESO::Particles {

class ParticleGroup;
class ParticleSubGroup;

/**
 * This is a mix-in class that provides EphemeralDat capability.
 */
class EphemeralDats {

  friend class ParticleGroup;
  friend class ParticleSubGroup;

protected:
  struct {
    SYCLTargetSharedPtr sycl_target;
    int ncell;
    std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> dats_real;
    std::map<Sym<INT>, ParticleDatSharedPtr<INT>> dats_int;
    INT npart_local{0};
    int *h_npart_cell{nullptr};
    int *d_npart_cell{nullptr};
    INT *d_npart_cell_es{nullptr};
    EventStack es;
  } ephemeral;

  EphemeralDats(SYCLTargetSharedPtr sycl_target, const int ncell) {
    this->ephemeral.sycl_target = sycl_target;
    this->ephemeral.ncell = ncell;
  }

  virtual inline void prepare_ephemeral_dats() = 0;

  inline void reset_ephemeral_dats(INT npart_local, int *h_npart_cell,
                                   int *d_npart_cell, INT *d_npart_cell_es) {
    this->ephemeral.es.wait();
    this->ephemeral.npart_local = npart_local;
    this->ephemeral.h_npart_cell = h_npart_cell;
    this->ephemeral.d_npart_cell = d_npart_cell;
    this->ephemeral.d_npart_cell_es = d_npart_cell_es;
    this->ephemeral.dats_real.clear();
    this->ephemeral.dats_int.clear();
  }

  inline void push_ephemeral_dat(Sym<REAL> sym,
                                 ParticleDatSharedPtr<REAL> dat) {
    this->ephemeral.dats_real[sym] = dat;
  }
  inline void push_ephemeral_dat(Sym<INT> sym, ParticleDatSharedPtr<INT> dat) {
    this->ephemeral.dats_int[sym] = dat;
  }
  inline void pop_ephemeral_dat(Sym<REAL> sym) {
    this->ephemeral.dats_real.erase(sym);
  }
  inline void pop_ephemeral_dat(Sym<INT> sym) {
    this->ephemeral.dats_int.erase(sym);
  }

  inline ParticleDatSharedPtr<REAL> get_ephemeral_dat(Sym<REAL> sym) const {
    NESOASSERT(this->contains_ephemeral_dat(sym),
               "Cannot find EphemeralDat with name: " + sym.name);
    return this->ephemeral.dats_real.at(sym);
  }

  inline ParticleDatSharedPtr<INT> get_ephemeral_dat(Sym<INT> sym) const {
    NESOASSERT(this->contains_ephemeral_dat(sym),
               "Cannot find EphemeralDat with name: " + sym.name);
    return this->ephemeral.dats_int.at(sym);
  }

public:
  virtual ~EphemeralDats() = default;
  /// Disable (implicit) copies.
  EphemeralDats(const EphemeralDats &st) = delete;
  /// Disable (implicit) copies.
  EphemeralDats &operator=(EphemeralDats const &a) = delete;

  /**
   * Add a new EphemeralDat by specifying the Sym and number of components.
   *
   * @param sym Sym<INT> or Sym<REAL> for new EphemeralDat.
   * @param int ncomp Number of components for the new EphemeralDat.
   */
  template <typename T>
  inline void add_ephemeral_dat(const Sym<T> sym, const int ncomp) {
    this->prepare_ephemeral_dats();
    NESOASSERT(0 < ncomp,
               "Bad number of components for EphemeralDat with name: " +
                   sym.name);
    NESOASSERT(!this->contains_ephemeral_dat(sym) ||
                   (this->contains_ephemeral_dat(sym) &&
                    this->get_ephemeral_dat(sym)->ncomp == ncomp),
               "EphemeralDat already exists with name: " + sym.name +
                   "but with a different number of components.");

    if (!this->contains_ephemeral_dat(sym)) {
      // This constructor is protected within ParticleDatT so we have to use a
      // new.
      auto ptr = std::shared_ptr<ParticleDatT<T>>(new ParticleDatT(
          this->ephemeral.sycl_target, sym, ncomp, this->ephemeral.ncell,
          this->ephemeral.npart_local, this->ephemeral.h_npart_cell,
          this->ephemeral.d_npart_cell, this->ephemeral.d_npart_cell_es));
      this->push_ephemeral_dat(sym, ptr);
    }
  }

  /**
   * Remove a EphemeralDat by specifying the Sym.
   *
   * @param sym Sym<INT> or Sym<REAL> for EphemeralDat to remove.
   */
  template <typename T> inline void remove_ephemeral_dat(const Sym<T> sym) {
    NESOASSERT(this->contains_ephemeral_dat(sym),
               "Cannot remove EphemeralDat with name: " + sym.name);
    this->ephemeral.es.wait();
    this->pop_ephemeral_dat(sym);
    NESOASSERT(!this->contains_ephemeral_dat(sym),
               "Failed to remove EphemeralDat with name: " + sym.name);
  }

  /**
   *  Determine if the Container contains a EphemeralDat of a given name.
   *
   *  @param sym Symbol of EphemeralDat.
   *  @returns True if EphemeralDat exists on this collection of EphemeralDats.
   */
  inline bool contains_ephemeral_dat(Sym<REAL> sym) const {
    return (bool)this->ephemeral.dats_real.count(sym);
  }

  /**
   *  Determine if the container contains a EphemeralDat of a given name.
   *
   *  @param sym Symbol of EphemeralDat.
   *  @returns True if EphemeralDat exists on this collection of EphemeralDats.
   */
  inline bool contains_ephemeral_dat(Sym<INT> sym) const {
    return (bool)this->ephemeral.dats_int.count(sym);
  }
};

} // namespace NESO::Particles

#endif
