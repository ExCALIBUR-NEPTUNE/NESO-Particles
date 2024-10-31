#ifndef _NESO_PARTICLES_PARTICLE_DAT
#define _NESO_PARTICLES_PARTICLE_DAT

#include <cmath>
#include <functional>
#include <limits>
#include <memory>

#include "access.hpp"
#include "cell_dat.hpp"
#include "compute_target.hpp"
#include "loop/access_descriptors.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

// forward declaration such that ParticleDat can define ParticleGroup as a
// friend class.
class ParticleGroup;
// Forward declaration of ParticleLoop such that LocalArray can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;
template <typename T> class SymVector;
class MeshHierarchyGlobalMap;
class CellMove;
class LayerCompressor;
class ParticlePacker;
class ParticleUnpacker;

template <typename T> using ParticleDatImplGetT = T ***;
template <typename T> using ParticleDatImplGetConstT = T *const *const *;
class ParticleGroup;

namespace ParticleLoopImplementation {

/**
 * Method to compute access to a particle dat (read) - via Sym.
 */
template <typename T>
inline ParticleDatImplGetConstT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<Sym<T> *> &a);
/**
 * Method to compute access to a particle dat (write) - via Sym
 */
template <typename T>
inline ParticleDatImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<Sym<T> *> &a);
/**
 * Method to compute access to a particle dat (read).
 */
template <typename T>
inline ParticleDatImplGetConstT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<ParticleDatT<T> *> &a);
/**
 * Method to compute access to a particle dat (write).
 */
template <typename T>
inline ParticleDatImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<ParticleDatT<T> *> &a);

} // namespace ParticleLoopImplementation

/**
 *  Wrapper around a CellDat to store particle data on a per cell basis.
 */
template <typename T> class ParticleDatT {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;
  friend class SymVector<T>;
  friend class ParticleGroup;
  friend class MeshHierarchyGlobalMap;
  friend class CellMove;
  friend class LayerCompressor;
  friend class ParticlePacker;
  friend class ParticleUnpacker;
  friend ParticleDatImplGetConstT<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<Sym<T> *> &a);
  friend ParticleDatImplGetT<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<Sym<T> *> &a);

  friend ParticleDatImplGetConstT<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<ParticleDatT<T> *> &a);
  friend ParticleDatImplGetT<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<ParticleDatT<T> *> &a);

private:
protected:
  std::function<void(const Sym<T> &, const int)> write_callback;

  inline void
  add_write_callback(std::function<void(const Sym<T> &, const int)> fn) {
    this->write_callback = fn;
  }

  inline void write_callback_wrapper(const int mode) {
    if (this->write_callback) {
      this->write_callback(this->sym, mode);
    }
  }

  /**
   * Non-const pointer to underlying device data. Intended for friend access
   * from ParticleLoop. These pointers must not be cached then used later
   * without recalling this function.
   */
  inline ParticleDatImplGetT<T> impl_get() {
    // write_callback is called in the cell dat
    return this->cell_dat.impl_get();
  }

  /**
   * Const pointer to underlying device data. Intended for friend access
   * from ParticleLoop. These pointers must not be cached then used later
   * without recalling this function.
   */
  inline ParticleDatImplGetConstT<T> impl_get_const() {
    return this->cell_dat.impl_get_const();
  }

  /// Pointer to the inclusive sum of cell occupancies computed by the
  /// ParticleGroup
  INT *d_npart_cell_es;
  inline void set_d_npart_cell_es(INT *ptr) { this->d_npart_cell_es = ptr; }
  inline INT *get_d_npart_cell_es() { return this->d_npart_cell_es; }

  inline void
  append_particle_data(std::shared_ptr<ParticleDatT<T>> particle_dat,
                       const std::vector<INT> &h_npart_cell_existing,
                       const std::vector<INT> &h_npart_cell_to_add,
                       EventStack &es);

  /**
   *  Const pointer to the start of a column in a cell. Not to be cached.
   */
  inline T const *col_device_const_ptr(const int cell, const int col) {
    return this->cell_dat.col_device_ptr(cell, col);
  }

public:
  /// Disable (implicit) copies.
  ParticleDatT(const ParticleDatT &st) = delete;
  /// Disable (implicit) copies.
  ParticleDatT &operator=(ParticleDatT const &a) = delete;

  /// Device only accessible array of the particle counts per cell.
  int *d_npart_cell;
  /// Host only accessible array of particle counts per cell.
  int *h_npart_cell;
  /// Sym object this ParticleDat was created with.
  const Sym<T> sym;
  /// CellDat instance that contains the actual particle data.
  CellDat<T> cell_dat;
  /// Number of components stored per particle (columns in the CellDat).
  const int ncomp;
  /// Number of cells this ParticleDat is defined on.
  const int ncell;
  /// Flat to indicate if this ParticleDat is to hold particle positions or
  // cell ids.
  const bool positions;
  /// Label given to the ParticleDat.
  const std::string name;
  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;

  /**
   * Create a new ParticleDat.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param sym Sym object that defines the type and label.
   * @param ncomp Number of components, of type defined in `sym`.
   * @param ncell Number of cells this ParticleDat is defined over.
   * @param positions Does this Dat hold particle positions or cell ids.
   */
  ParticleDatT(SYCLTargetSharedPtr sycl_target, const Sym<T> sym, int ncomp,
               int ncell, bool positions = false)
		: sym(sym), cell_dat(CellDat<T>(sycl_target, ncell, ncomp)), 
		ncomp(ncomp), ncell(ncell),   
		positions(positions),name(sym.name), sycl_target(sycl_target)
	{

    this->cell_dat.add_write_callback(std::bind(
        &ParticleDatT<T>::write_callback_wrapper, this, std::placeholders::_1));

    this->h_npart_cell =
        sycl::malloc_host<int>(this->ncell, this->sycl_target->queue);
    this->d_npart_cell =
        sycl::malloc_device<int>(this->ncell, this->sycl_target->queue);
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell[cellx] = 0;
    }
    this->npart_host_to_device();
  }

  /**
   *  Copy cell particle counts from host buffer to device buffer.
   */
  inline void npart_host_to_device() {
    if (this->ncell > 0) {
      this->sycl_target->queue
          .memcpy(this->d_npart_cell, this->h_npart_cell,
                  this->ncell * sizeof(int))
          .wait();
    }
  }
  /**
   *  Asynchronously copy cell particle counts from host buffer to device
   * buffer.
   *
   *  @returns sycl::event for copy operation.
   */
  inline sycl::event async_npart_host_to_device() {
    NESOASSERT(this->ncell > 0, "Zero sized memcpy issued");
    return this->sycl_target->queue.memcpy(
        this->d_npart_cell, this->h_npart_cell, this->ncell * sizeof(int));
  }
  /**
   *  Copy cell particle counts from device buffer to host buffer.
   */
  inline void npart_device_to_host() {
    if (this->ncell > 0) {
      this->sycl_target->queue
          .memcpy(this->h_npart_cell, this->d_npart_cell,
                  this->ncell * sizeof(int))
          .wait();
    }
  }
  /**
   *  Asynchronously copy cell particle counts from device buffer to host
   * buffer.
   *
   *  @returns sycl::event for copy operation.
   */
  inline sycl::event async_npart_device_to_host() {
    NESOASSERT(this->ncell > 0, "Zero sized memcpy issued");
    return this->sycl_target->queue.memcpy(
        this->h_npart_cell, this->d_npart_cell, this->ncell * sizeof(int));
  }

  ~ParticleDatT() {
    sycl::free(this->h_npart_cell, this->sycl_target->queue);
    sycl::free(this->d_npart_cell, this->sycl_target->queue);
  }

  /**
   *  Add particle data to the ParticleDat.
   *
   *  @param npart_new Number of new particles to add.
   *  @param new_data_exists Indicate if there is new data to copy of if the
   *  data should be initialised with zeros.
   *  @param cells Cell indices of the new particles.
   *  @param layers Layer (row) indices of the new particles.
   *  @param data Particle data to copy into the ParticleDat.
   *  @param es EventStack to push events onto.
   */
  inline void append_particle_data(const int npart_new,
                                   const bool new_data_exists,
                                   std::vector<INT> &cells,
                                   std::vector<INT> &layers,
                                   std::vector<T> &data, EventStack &es);

  /**
   *  Realloc the underlying CellDat such that the indicated new number of
   *  particles can be stored.
   *
   *  @param npart_cell_new Particle counts for each cell.
   */
  inline void realloc(std::vector<INT> &npart_cell_new);
  /**
   *  Realloc the underlying CellDat such that the indicated new number of
   *  particles can be stored.
   *
   *  @param npart_cell_new Particle counts for each cell.
   */
  template <typename U> inline void realloc(BufferShared<U> &npart_cell_new);
  /**
   *  Realloc the underlying CellDat such that the indicated new number of
   *  particles can be stored.
   *
   *  @param npart_cell_new Particle counts for each cell.
   */
  template <typename U> inline void realloc(BufferHost<U> &npart_cell_new);
  /**
   *  Realloc the underlying CellDat such that the indicated new number of
   *  particles can be stored for a particular cell.
   *
   *  @param cell Cell index to reallocate.
   *  @param npart_cell_new Particle counts for the cell.
   */
  inline void realloc(const int cell, const int npart_cell_new);

  /**
   * Asynchronously copy particle data from old cells/layers to new
   * cells/layers.
   *
   * @param npart Number of particles to copy.
   * @param d_cells_old Device pointer to an array of old cells.
   * @param d_cells_new Device pointer to an array of new cells.
   * @param d_layers_old Device pointer to an array of old layers.
   * @param d_layers_new Device pointer to an array of new layers.
   * @returns sycl::event to wait on for data copy.
   */
  inline sycl::event copy_particle_data(const int npart, 
										[[maybe_unused]] const INT *d_cells_old,
                                        [[maybe_unused]] const INT *d_cells_new,
                                        const INT *d_layers_old,
                                        const INT *d_layers_new) {
    this->write_callback_wrapper(0);
    const size_t npart_s = static_cast<size_t>(npart);
    T ***d_cell_dat_ptr = this->impl_get();
    const int ncomp = this->ncomp;

    sycl::event event =
        this->sycl_target->queue.submit([&](sycl::handler &cgh) {
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

  /**
   * Get the number of particles stored in all cells
   *
   * @returns Total number of stored particles.
   */
  inline INT get_npart_local() {
    INT n = 0;
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      n += h_npart_cell[cellx];
    }
    return n;
  }

  /**
   * Async call to set d_npart_cells from a device buffer. npart_device_to_host
   * must be called on event completion.
   *
   * @param d_npart_cell_in Device accessible pointer to an array containing new
   * cell counts.
   * @returns sycl::event to wait on for completion.
   */
  template <typename U>
  inline sycl::event set_npart_cells_device(const U *d_npart_cell_in) {
    this->write_callback_wrapper(0);
    const size_t ncell = static_cast<size_t>(this->ncell);
    int *k_npart_cell = this->d_npart_cell;
    sycl::event event =
        this->sycl_target->queue.submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(ncell), [=](sycl::id<1> idx) {
            k_npart_cell[idx] = static_cast<int>(d_npart_cell_in[idx]);
          });
        });
    return event;
  }

  /**
   *  Set cell particle counts from host accessible pointer.
   *  npart_host_to_device must be called to set counts on the device.
   *
   *  @param h_npart_cell_in Host pointer to cell particle counts.
   */
  template <typename U>
  inline void set_npart_cells_host(const U *h_npart_cell_in) {
    this->write_callback_wrapper(0);
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell[cellx] = h_npart_cell_in[cellx];
    }
  }

  /**
   *  Set the particle count in a single cell. Assigns on both host and device.
   *
   *  @param cell Cell to set particle count for.
   *  @param npart New particle count for cell.
   */
  inline void set_npart_cell(const INT cell, const int npart) {
    this->write_callback_wrapper(0);
    this->h_npart_cell[cell] = npart;
    this->sycl_target->queue
        .memcpy(this->d_npart_cell + cell, this->h_npart_cell + cell,
                sizeof(int))
        .wait();
    return;
  }
  /**
   *  Set particle counts in cells from std::vector. Sets on both host and
   * device.
   *
   *  @param npart std::vector of new particle counts per cell.
   */
  inline void set_npart_cells(std::vector<INT> &npart) {
    this->write_callback_wrapper(0);
    NESOASSERT(npart.size() >= this->ncell, "bad vector size");
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell[cellx] = npart[cellx];
    }
    this->npart_host_to_device();
    return;
  }

  /**
   *  Set the particle counts in cells from a BufferHost. Sets on both host and
   *  device.
   *
   *  @param h_npart_cell_in New particle counts per cell.
   */
  template <typename U>
  inline void set_npart_cells(const BufferHost<U> &h_npart_cell_in) {
    this->write_callback_wrapper(0);
    NESOASSERT(h_npart_cell_in.size >= this->ncell, "bad BufferHost size");
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell[cellx] = h_npart_cell_in.ptr[cellx];
    }
    this->npart_host_to_device();
    return;
  }

  /**
   *  Asynchronously set the particle counts in cells from a BufferHost. Sets
   *  on both host and device.
   *
   *  @param h_npart_cell_in New particle counts per cell.
   *  @returns sycl::event to wait on for device assignment.
   */
  template <typename U>
  inline sycl::event
  async_set_npart_cells(const BufferHost<U> &h_npart_cell_in) {
    this->write_callback_wrapper(0);
    NESOASSERT(h_npart_cell_in.size >= static_cast<std::size_t>(this->ncell), "bad BufferHost size");
    for (int cellx = 0; cellx < this->ncell; cellx++) {
      this->h_npart_cell[cellx] = h_npart_cell_in.ptr[cellx];
    }
    return this->async_npart_host_to_device();
  }

  /**
   *  Reduce the row count of the underlying CellDat to the per cell particle
   *  occupancies.
   */
  inline void trim_cell_dat_rows();

  /**
   *  Utility function to print the contents of all or a select range of cells.
   *
   *  @param start Optional first cell to start printing from.
   *  @param end Optional end cell +1 to end printing at.
   */
  inline void print(const int start = 0, int end = -1) {
    if (end < 0) {
      end = this->ncell;
    }
    std::cout << this->sym.name << std::endl;
    this->cell_dat.print(start, end);
  };

  /**
   *  Get an upper bound for the number of particles. May be a very bad
   *  approximation.
   *
   *  @returns Upper bound for number of particles.
   */
  inline INT get_npart_upper_bound() {
    return this->cell_dat.ncells * this->cell_dat.get_nrow_max();
  }

  /**
   *  Get the size of the iteration set required to loop over all particles
   *  with the particle loop macros.
   *
   *  @returns Particle Loop iteration set size.
   */
  inline INT get_particle_loop_iter_range() {
    // #ifdef NESO_PARTICLES_ITER_PARTICLES
    //     return this->cell_dat.ncells * this->cell_dat.get_nrow_max();
    // #else // case for NESO_PARTICLES_ITER_CELLS
    //     const auto n = this->cell_dat.get_nrow_max();
    //     const auto m = (n/NESO_PARTICLES_BLOCK_SIZE) + ((INT) (n %
    //     NESO_PARTICLES_BLOCK_SIZE > 0)); return this->cell_dat.ncells * m;
    // #endif
    //

    INT iter_range =
        this->cell_dat.ncells * this->get_particle_loop_cell_stride();

    // The scyl range takes a size_t but the implementations fail with larger
    // than int arguments.
    NESOASSERT(iter_range >= 0, "Negative iter_range?");
    NESOASSERT(iter_range <= std::numeric_limits<int>::max(),
               "ParticleLoop iter range exceeds int limits.");

    return iter_range;
  }
  /**
   *  Get the size of the iteration set per cell stride required to loop over
   *  all particles with the particle loop macros.
   *
   *  @returns Particle Loop iteration stride size.
   */
  inline INT get_particle_loop_cell_stride() {

#ifdef NESO_PARTICLES_ITER_PARTICLES
    return this->cell_dat.get_nrow_max();
#else
    const INT n = this->cell_dat.get_nrow_max();
    const INT m = (n / NESO_PARTICLES_BLOCK_SIZE) +
                  ((INT)(n % NESO_PARTICLES_BLOCK_SIZE > 0));
    return m;
#endif
  }
  /**
   *  Get a device pointer to the array that stores particle counts per cell
   *  for the particle loop macros.
   *
   *  @returns Device pointer to particle counts per cell.
   */
  inline int *get_particle_loop_npart_cell() { return this->d_npart_cell; }

  /**
   *  Wait for a realloc to complete.
   */
  inline void wait_realloc() { this->cell_dat.wait_set_nrow(); }
};

template <typename T>
using ParticleDatSharedPtr = std::shared_ptr<ParticleDatT<T>>;

template <typename T>
inline ParticleDatSharedPtr<T> ParticleDat(SYCLTargetSharedPtr sycl_target,
                                           const Sym<T> sym, int ncomp,
                                           int ncell, bool positions = false) {
  return std::make_shared<ParticleDatT<T>>(sycl_target, sym, ncomp, ncell,
                                           positions);
}
template <typename T>
inline ParticleDatSharedPtr<T> ParticleDat(SYCLTargetSharedPtr sycl_target,
                                           ParticleProp<T> prop, int ncell) {
  return std::make_shared<ParticleDatT<T>>(sycl_target, prop.sym, prop.ncomp,
                                           ncell, prop.positions);
}

template <typename T>
inline void ParticleDatT<T>::realloc(std::vector<INT> &npart_cell_new) {
  NESOASSERT(npart_cell_new.size() >= static_cast<std::size_t>(this->ncell),
             "Insufficent new cell counts");
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->cell_dat.set_nrow(cellx, npart_cell_new[cellx]);
  }
}
template <typename T>
template <typename U>
inline void ParticleDatT<T>::realloc(BufferShared<U> &npart_cell_new) {
  NESOASSERT(npart_cell_new.size >= static_cast<std::size_t>(this->ncell), "Insufficent new cell counts");
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->cell_dat.set_nrow(cellx, npart_cell_new.ptr[cellx]);
  }
}
template <typename T>
template <typename U>
inline void ParticleDatT<T>::realloc(BufferHost<U> &npart_cell_new) {
  NESOASSERT(npart_cell_new.size >= static_cast<std::size_t>(this->ncell), "Insufficent new cell counts");
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    this->cell_dat.set_nrow_inner(cellx, npart_cell_new.ptr[cellx]);
  }
}
template <typename T>
inline void ParticleDatT<T>::realloc(const int cell, const int npart_cell_new) {
  NESOASSERT(npart_cell_new >= 0, "Bad cell new cell npart.");
  this->cell_dat.set_nrow(cell, npart_cell_new);
}

template <typename T> inline void ParticleDatT<T>::trim_cell_dat_rows() {
  this->cell_dat.reduce_nrow(this->h_npart_cell);
  this->write_callback_wrapper(0);
}

/*
 *  Append particle data to the ParticleDat. wait() must be called on the
 * event_stack before use of the data. Assumes the dat has already been
 * reallocated.
 */
template <typename T>
inline void ParticleDatT<T>::append_particle_data(
    const int npart_new, const bool new_data_exists, std::vector<INT> &cells,
    std::vector<INT> &layers, std::vector<T> &data, EventStack &es) {

  if (npart_new == 0) {
    return;
  }

  this->write_callback_wrapper(0);

  NESOASSERT(static_cast<std::size_t>(npart_new) <= cells.size(), "incorrect number of cells");

  // using "this" in the kernel causes segfaults on the device so we make a
  // copy here.
  const size_t size_npart_new = static_cast<size_t>(npart_new);
  const int ncomp = this->ncomp;
  T ***d_cell_dat_ptr = this->impl_get();

  sycl::buffer<INT, 1> b_cells(cells.data(), sycl::range<1>{size_npart_new});
  sycl::buffer<INT, 1> b_layers(layers.data(), sycl::range<1>{size_npart_new});

  // If data is supplied copy the data otherwise zero the components.
  if (new_data_exists) {
    sycl::buffer<T, 1> b_data(data.data(),
                              sycl::range<1>{size_npart_new * this->ncomp});
    es.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
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
    }));
  } else {
    es.push(this->sycl_target->queue.submit([&](sycl::handler &cgh) {
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
    }));
  }

  for (int px = 0; px < npart_new; px++) {
    auto cellx = cells[px];
    this->h_npart_cell[cellx]++;
  }
  es.push(this->async_npart_host_to_device());
}
/*
 *  Append particle data to the ParticleDat. wait() must be called on the
 * event_stack before use of the data. Assumes the dat has already been
 * reallocated.
 */
template <typename T>
inline void ParticleDatT<T>::append_particle_data(
    ParticleDatSharedPtr<T> particle_dat,
    const std::vector<INT> &h_npart_cell_existing,
    const std::vector<INT> &h_npart_cell_to_add, EventStack &es) {

  if (particle_dat != nullptr) {
    NESOASSERT(
        this->ncell == particle_dat->ncell,
        "Source and destination ParticleDats have different cell counts");
    NESOASSERT(this->ncomp == particle_dat->ncomp,
               "Source and destination ParticleDats have different number of "
               "components");
  }

  this->write_callback_wrapper(0);
  for (int cellx = 0; cellx < this->ncell; cellx++) {
    const INT layer_start = h_npart_cell_existing.at(cellx);
    const INT layer_to_add = h_npart_cell_to_add.at(cellx);
    if (layer_to_add > 0) {

      for (int colx = 0; colx < this->ncomp; colx++) {
        auto d_ptr_dst = this->cell_dat.col_device_ptr(cellx, colx);
        NESOASSERT(d_ptr_dst != nullptr, "Bad dst ptr.");
        if (particle_dat == nullptr) {
          es.push(this->sycl_target->queue.fill(
              d_ptr_dst + layer_start, static_cast<T>(0), layer_to_add));
        } else {
          const auto d_ptr_src =
              particle_dat->cell_dat.col_device_ptr(cellx, colx);
          NESOASSERT(d_ptr_src != nullptr, "Bad src ptr.");
          es.push(this->sycl_target->queue.memcpy(
              d_ptr_dst + layer_start, d_ptr_src, layer_to_add * sizeof(T)));
        }
      }
    }
  }
}

} // namespace NESO::Particles

#endif
