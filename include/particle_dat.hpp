#ifndef _NESO_PARTICLES_PARTICLE_DAT
#define _NESO_PARTICLES_PARTICLE_DAT

#include <CL/sycl.hpp>
#include <memory>

#include "access.hpp"
#include "compute_target.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

template <typename T> class ParticleDatT {
private:
  int npart_local;
  int npart_alloc;

public:
  int *s_npart_cell;
  const Sym<T> sym;
  CellDat<T> cell_dat;
  const int ncomp;
  const int ncell;
  const bool positions;
  const std::string name;

  SYCLTarget &sycl_target;

  ParticleDatT(SYCLTarget &sycl_target, const Sym<T> sym, int ncomp, int ncell,
               bool positions = false)
      : sycl_target(sycl_target), sym(sym), name(sym.name), ncomp(ncomp),
        ncell(ncell), positions(positions),
        cell_dat(CellDat<T>(sycl_target, ncell, ncomp)) {

    this->npart_local = 0;
    this->npart_alloc = 0;

    this->s_npart_cell =
        sycl::malloc_shared<int>(this->ncell, this->sycl_target.queue);
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->s_npart_cell[cellx] = 0;
    }
  }
  ~ParticleDatT() { sycl::free(this->s_npart_cell, this->sycl_target.queue); }

  inline void set_compute_target(SYCLTarget &sycl_target) {
    this->sycl_target = sycl_target;
  }
  inline int get_npart_local(const int npart_local) {
    return this->npart_local;
  }
  inline void append_particle_data(const int npart_new,
                                   const bool new_data_exists,
                                   std::vector<INT> &cells,
                                   std::vector<T> &data);
  inline void realloc(std::vector<INT> &npart_cell_new);
  inline int get_npart_local() { return this->npart_local; }
};

template <typename T> using ParticleDatShPtr = std::shared_ptr<ParticleDatT<T>>;

template <typename T>
inline ParticleDatShPtr<T> ParticleDat(SYCLTarget &sycl_target,
                                       const Sym<T> sym, int ncomp, int ncell,
                                       bool positions = false) {
  return std::make_shared<ParticleDatT<T>>(sycl_target, sym, ncomp, ncell,
                                           positions);
}
template <typename T>
inline ParticleDatShPtr<T> ParticleDat(SYCLTarget &sycl_target,
                                       ParticleProp<T> prop, int ncell) {
  return std::make_shared<ParticleDatT<T>>(sycl_target, prop.sym, prop.ncomp,
                                           ncell, prop.positions);
}
template <typename T>
inline void ParticleDatT<T>::realloc(std::vector<INT> &npart_cell_new) {
  NESOASSERT(npart_cell_new.size() >= this->ncell,
             "Insufficent new cell counts");
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->cell_dat.set_nrow(cellx, npart_cell_new[cellx]);
  }
}

/*
 *  Append particle data to the ParticleDat. wait() must be called on the queue
 *  before use of the data.
 *
 */
template <typename T>
inline void ParticleDatT<T>::append_particle_data(const int npart_new,
                                                  const bool new_data_exists,
                                                  std::vector<INT> &cells,
                                                  std::vector<T> &data) {

  NESOASSERT(npart_new <= cells.size(), "incorrect number of cells");

  // using "this" in the kernel causes segfaults on the device so we make a
  // copy here.
  const size_t size_npart_new = static_cast<size_t>(npart_new);
  int *s_npart_cell = this->s_npart_cell;
  const int ncomp = this->ncomp;
  T ***d_cell_dat_ptr = this->cell_dat.device_ptr();

  sycl::buffer<INT, 1> b_cells(cells.data(), sycl::range<1>{size_npart_new});

  // If data is supplied copy the data otherwise zero the components.
  if (new_data_exists) {
    sycl::buffer<T, 1> b_data(data.data(),
                              sycl::range<1>{size_npart_new * this->ncomp});
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      // The cell counts on this dat
      auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);
      // The new data
      auto a_data = b_data.template get_access<sycl::access::mode::read>(cgh);

      cgh.parallel_for<>(sycl::range<1>(npart_new), [=](sycl::id<1> idx) {
        INT cellx = a_cells[idx];
        // atomically get the new layer and increment the count in
        // the cell
#if defined(__INTEL_LLVM_COMPILER)
        auto element_atomic = sycl::ext::oneapi::atomic_ref<
            int, sycl::ext::oneapi::memory_order_acq_rel,
            sycl::ext::oneapi::memory_scope_device,
            sycl::access::address_space::global_space>(s_npart_cell[cellx]);
        const int layerx = element_atomic.fetch_add(1);
#else
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        element_atomic(s_npart_cell[cellx]);
                    const int layerx = element_atomic.fetch_add(1);
#endif
        // copy the data into the dat.
        for (int cx = 0; cx < ncomp; cx++) {
          d_cell_dat_ptr[cellx][cx][layerx] = a_data[cx * npart_new + idx];
        }
      });
    });
  } else {
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      // The cell counts on this dat
      auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);

      cgh.parallel_for<>(sycl::range<1>(npart_new), [=](sycl::id<1> idx) {
        INT cellx = a_cells[idx];
        // atomically get the layer

#if defined(__INTEL_LLVM_COMPILER)
        auto element_atomic = sycl::ext::oneapi::atomic_ref<
            int, sycl::ext::oneapi::memory_order_acq_rel,
            sycl::ext::oneapi::memory_scope_device,
            sycl::access::address_space::global_space>(s_npart_cell[cellx]);
        const int layerx = element_atomic.fetch_add(1);
#else
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        element_atomic(s_npart_cell[cellx]);
                    const int layerx = element_atomic.fetch_add(1);
#endif

        // zero the new components in the dat
        for (int cx = 0; cx < ncomp; cx++) {
          d_cell_dat_ptr[cellx][cx][layerx] = ((T)0);
        }
      });
    });
  }
}

} // namespace NESO::Particles

#endif
