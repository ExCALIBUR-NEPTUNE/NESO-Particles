#ifndef _NESO_PARTICLES_PARTICLE_DAT
#define _NESO_PARTICLES_PARTICLE_DAT

#include <CL/sycl.hpp>
#include <memory>

#include "access.hpp"
#include "cell_dat.hpp"
#include "compute_target.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"

using namespace cl;

namespace NESO::Particles {

template <typename T> class ParticleDatT {
private:
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
  inline void append_particle_data(const int npart_new,
                                   const bool new_data_exists,
                                   std::vector<INT> &cells,
                                   std::vector<INT> &layers,
                                   std::vector<T> &data);
  inline void realloc(std::vector<INT> &npart_cell_new);
  inline void realloc(BufferShared<INT> &npart_cell_new);
  inline void realloc(const int cell, const int npart_cell_new);

  inline sycl::event copy_particle_data(const int npart, const INT *d_cells_old,
                                        const INT *d_cells_new,
                                        const INT *d_layers_old,
                                        const INT *d_layers_new) {
    const size_t npart_s = static_cast<size_t>(npart);
    T ***d_cell_dat_ptr = this->cell_dat.device_ptr();
    const int ncomp = this->ncomp;

    sycl::event event = this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<1>(npart_s), [=](sycl::id<1> idx) {
        const INT cell_oldx = d_cells_old[idx];
        // remove particles currently masks of elements using -1
        if (cell_oldx > -1) {
          const INT cell_newx = d_cells_old[idx];
          const INT layer_oldx = d_layers_old[idx];
          const INT layer_newx = d_layers_new[idx];

          // copy the data from old cell/layer to new cell/layer.
          for (int cx = 0; cx < ncomp; cx++) {
            d_cell_dat_ptr[cell_newx][cx][layer_newx] =
                d_cell_dat_ptr[cell_oldx][cx][layer_oldx];
          }
        }
      });
    });

    return event;
  }

  /*
   * Async call to set s_npart_cells from a device buffer
   */
  template <typename U>
  inline sycl::event set_npart_cells_device(const U *d_npart_cell) {
    const size_t ncell = static_cast<size_t>(this->ncell);
    int *s_npart_cell_ptr = this->s_npart_cell;
    sycl::event event = this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<1>(ncell), [=](sycl::id<1> idx) {
        s_npart_cell_ptr[idx] = static_cast<int>(d_npart_cell[idx]);
      });
    });
    return event;
  }

  inline void set_npart_cell(const INT cell, const int npart) {
    s_npart_cell[cell] = npart;
    return;
  }
  inline void set_npart_cells(std::vector<INT> &npart) {
    NESOASSERT(npart.size() >= this->ncell, "bad vector size");
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      s_npart_cell[cellx] = npart[cellx];
    }
    return;
  }
  inline void trim_cell_dat_rows();
  inline void print(const int start = 0, int end = -1) {
    if (end < 0) {
      end = this->ncell;
    }
    std::cout << this->sym.name << std::endl;
    this->cell_dat.print(start, end);
  };

  inline INT get_particle_loop_iter_range() {
    return this->cell_dat.ncells * this->cell_dat.get_nrow_max();
  }
  inline INT get_particle_loop_cell_stride() {
    return this->cell_dat.get_nrow_max();
  }
  inline int *get_particle_loop_npart_cell() { return this->s_npart_cell; }
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
  this->cell_dat.compute_nrow_max();
}
template <typename T>
inline void ParticleDatT<T>::realloc(BufferShared<INT> &npart_cell_new) {
  NESOASSERT(npart_cell_new.size >= this->ncell, "Insufficent new cell counts");
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->cell_dat.set_nrow(cellx, npart_cell_new.ptr[cellx]);
  }
  this->cell_dat.compute_nrow_max();
}

template <typename T>
inline void ParticleDatT<T>::realloc(const int cell, const int npart_cell_new) {
  NESOASSERT(npart_cell_new >= 0, "Bad cell new cell npart.");
  this->cell_dat.set_nrow(cell, npart_cell_new);
  this->cell_dat.compute_nrow_max();
}

template <typename T> inline void ParticleDatT<T>::trim_cell_dat_rows() {
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->cell_dat.set_nrow(cellx, s_npart_cell[cellx]);
  }
  this->cell_dat.compute_nrow_max();
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
                                                  std::vector<INT> &layers,
                                                  std::vector<T> &data) {

  NESOASSERT(npart_new <= cells.size(), "incorrect number of cells");

  // using "this" in the kernel causes segfaults on the device so we make a
  // copy here.
  const size_t size_npart_new = static_cast<size_t>(npart_new);
  int *s_npart_cell = this->s_npart_cell;
  const int ncomp = this->ncomp;
  T ***d_cell_dat_ptr = this->cell_dat.device_ptr();

  sycl::buffer<INT, 1> b_cells(cells.data(), sycl::range<1>{size_npart_new});
  sycl::buffer<INT, 1> b_layers(layers.data(), sycl::range<1>{size_npart_new});

  // If data is supplied copy the data otherwise zero the components.
  if (new_data_exists) {
    sycl::buffer<T, 1> b_data(data.data(),
                              sycl::range<1>{size_npart_new * this->ncomp});
    this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      // The cell counts on this dat
      auto a_cells = b_cells.get_access<sycl::access::mode::read>(cgh);
      auto a_layers = b_layers.get_access<sycl::access::mode::read>(cgh);
      // The new data
      auto a_data = b_data.template get_access<sycl::access::mode::read>(cgh);
      cgh.parallel_for<>(sycl::range<1>(npart_new), [=](sycl::id<1> idx) {
        const INT cellx = a_cells[idx];
        const INT layerx = a_layers[idx];
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
      auto a_layers = b_layers.get_access<sycl::access::mode::read>(cgh);

      cgh.parallel_for<>(sycl::range<1>(npart_new), [=](sycl::id<1> idx) {
        const INT cellx = a_cells[idx];
        const INT layerx = a_layers[idx];
        // zero the new components in the dat
        for (int cx = 0; cx < ncomp; cx++) {
          d_cell_dat_ptr[cellx][cx][layerx] = ((T)0);
        }
      });
    });
  }

  for (int px = 0; px < npart_new; px++) {
    auto cellx = cells[px];
    s_npart_cell[cellx]++;
  }
}

} // namespace NESO::Particles

#endif
