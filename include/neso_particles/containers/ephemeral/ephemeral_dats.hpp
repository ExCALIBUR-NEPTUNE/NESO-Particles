#ifndef __NESO_PARTICLES_EPHEMERAL_EPHEMERAL_DATS_HPP_
#define __NESO_PARTICLES_EPHEMERAL_EPHEMERAL_DATS_HPP_

#include "ephemeral_dat.hpp"

namespace NESO::Particles {

class ParticleGroup;
class ParticleSubGroup;

class EphemeralDats {

  friend class ParticleGroup;
  friend class ParticleSubGroup;

protected:
  SYCLTargetSharedPtr sycl_target;
  int ncell;
  std::map<Sym<REAL>, EphemeralDatSharedPtr<REAL>> dats_real;
  std::map<Sym<INT>, EphemeralDatSharedPtr<INT>> dats_int;

  EphemeralDats(SYCLTargetSharedPtr sycl_target, const int ncell)
      : sycl_target(sycl_target), ncell(ncell) {}

  INT npart_local{0};
  INT *d_npart_cell_es{nullptr};
  EventStack es;

  inline void reset(const INT npart_local, INT *d_npart_cell_es) {
    this->es.wait();
    this->npart_local = npart_local;
    this->d_npart_cell_es = d_npart_cell_es;
    this->dats_real.clear();
    this->dats_int.clear();
  }

  inline void push(Sym<REAL> sym, EphemeralDatSharedPtr<REAL> dat) {
    this->dats_real[sym] = dat;
  }
  inline void push(Sym<INT> sym, EphemeralDatSharedPtr<INT> dat) {
    this->dats_int[sym] = dat;
  }
  inline void pop(Sym<REAL> sym) { this->dats_real.erase(sym); }
  inline void pop(Sym<INT> sym) { this->dats_int.erase(sym); }

  inline EphemeralDatSharedPtr<REAL> get_ephemeral_dat(Sym<REAL> sym) const {
    NESOASSERT(this->contains_ephemeral_dat(sym),
               "Cannot find EphemeralDat with name: " + sym.name);
    return this->dats_real.at(sym);
  }

  inline EphemeralDatSharedPtr<INT> get_ephemeral_dat(Sym<INT> sym) const {
    NESOASSERT(this->contains_ephemeral_dat(sym),
               "Cannot find EphemeralDat with name: " + sym.name);
    return this->dats_int.at(sym);
  }

public:
  ~EphemeralDats() = default;
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
    NESOASSERT(0 < ncomp,
               "Bad number of components for EphemeralDat with name: " +
                   sym.name);
    NESOASSERT(!this->contains_ephemeral_dat(sym) ||
                   (this->contains_ephemeral_dat(sym) &&
                    this->get_ephemeral_dat(sym)->ncomp == ncomp),
               "EphemeralDat already exists with name: " + sym.name +
                   "but with a different number of components.");

    if (!this->contains_ephemeral_dat(sym)) {
      auto d_cell_ptrs =
          std::make_shared<BufferDevice<T **>>(this->sycl_target, this->ncell);
      auto d_col_ptrs = std::make_shared<BufferDevice<T *>>(
          this->sycl_target, this->ncell * ncomp);
      auto d_data = std::make_shared<BufferDevice<T>>(
          this->sycl_target, this->npart_local * ncomp);
      this->push(sym, std::make_shared<EphemeralDat<T>>(d_cell_ptrs, d_col_ptrs,
                                                        d_data));

      auto k_cell_ptrs = d_cell_ptrs->ptr;
      auto k_col_ptrs = d_col_ptrs->ptr;
      auto k_data = d_data->ptr;

      auto k_npart_cell_es = this->d_npart_cell_es;
      auto k_ncell = this->ncell;
      auto k_npart_local = this->npart_local;

      es.push(this->sycl_target->queue.parallel_for(
          sycl::range<1>(k_ncell), [=](auto cellx) {
            const INT npart_cell =
                (cellx < (k_ncell - 1))
                    ? k_npart_cell_es[cellx + 1] - k_npart_cell_es[cellx]
                    : k_npart_local - k_npart_cell_es[cellx];

            const INT npart_cell_es = k_npart_cell_es[cellx];
            T *base_ptr = k_data + npart_cell_es * ncomp;
            for (int cx = 0; cx < ncomp; cx++) {
              auto col_ptr = base_ptr + cx * npart_cell;
              k_col_ptrs[cellx * ncomp + cx] = col_ptr;
            }
            k_cell_ptrs[cellx] = k_col_ptrs + cellx * ncomp;
          }));
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
    this->es.wait();
    this->pop(sym);
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
    return (bool)this->dats_real.count(sym);
  }

  /**
   *  Determine if the container contains a EphemeralDat of a given name.
   *
   *  @param sym Symbol of EphemeralDat.
   *  @returns True if EphemeralDat exists on this collection of EphemeralDats.
   */
  inline bool contains_ephemeral_dat(Sym<INT> sym) const {
    return (bool)this->dats_int.count(sym);
  }
};

} // namespace NESO::Particles

#endif
