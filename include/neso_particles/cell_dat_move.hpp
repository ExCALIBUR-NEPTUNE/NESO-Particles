#ifndef _NESO_PARTICLES_CELL_DAT_MOVE
#define _NESO_PARTICLES_CELL_DAT_MOVE

#include <CL/sycl.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "cell_dat.hpp"
#include "cell_dat_compression.hpp"
#include "compute_target.hpp"
#include "error_propagate.hpp"
#include "particle_dat.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/**
 *  CellMove is the implementation that moves particles between cells. When a
 *  particle moves to a new cell the corresponding data is moved to the new
 *  cell in all the ParticleDat instances.
 */
class CellMove {
private:
  const int ncell;

  ParticleDatSharedPtr<INT> cell_id_dat;
  std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real;
  std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int;

  BufferHost<int> h_npart_cell;
  BufferDevice<int> d_npart_cell;

  // Buffers to store the old/new cells/layers
  BufferDevice<int> d_cells_old;
  BufferDevice<int> d_cells_new;
  BufferDevice<int> d_layers_old;
  BufferDevice<int> d_layers_new;

  // move count buffers;
  BufferHost<int> h_move_count;
  BufferDevice<int> d_move_count;

  // members for ParticleDat interface
  int num_dats_real = 0;
  int num_dats_int = 0;
  // host buffers
  BufferHost<REAL ***> h_particle_dat_ptr_real;
  BufferHost<INT ***> h_particle_dat_ptr_int;
  BufferHost<int> h_particle_dat_ncomp_real;
  BufferHost<int> h_particle_dat_ncomp_int;
  // device buffers
  BufferDevice<REAL ***> d_particle_dat_ptr_real;
  BufferDevice<INT ***> d_particle_dat_ptr_int;
  BufferDevice<int> d_particle_dat_ncomp_real;
  BufferDevice<int> d_particle_dat_ncomp_int;

  // layer compressor from the ParticleGroup for removing the old particle rows
  LayerCompressor &layer_compressor;

  // ErrorPropagate object to detect bad cell indices
  ErrorPropagate ep_bad_cell_indices;

  inline void get_particle_dat_info() {

    this->num_dats_real = this->particle_dats_real.size();
    this->h_particle_dat_ptr_real.realloc_no_copy(this->num_dats_real);
    this->h_particle_dat_ncomp_real.realloc_no_copy(this->num_dats_real);
    this->d_particle_dat_ptr_real.realloc_no_copy(this->num_dats_real);
    this->d_particle_dat_ncomp_real.realloc_no_copy(this->num_dats_real);

    this->num_dats_int = this->particle_dats_int.size();
    this->h_particle_dat_ptr_int.realloc_no_copy(this->num_dats_int);
    this->h_particle_dat_ncomp_int.realloc_no_copy(this->num_dats_int);
    this->d_particle_dat_ptr_int.realloc_no_copy(this->num_dats_int);
    this->d_particle_dat_ncomp_int.realloc_no_copy(this->num_dats_int);

    int index = 0;
    for (auto &dat : this->particle_dats_real) {
      this->h_particle_dat_ptr_real.ptr[index] = dat.second->impl_get();
      this->h_particle_dat_ncomp_real.ptr[index] = dat.second->ncomp;
      index++;
    }
    index = 0;
    for (auto &dat : particle_dats_int) {
      this->h_particle_dat_ptr_int.ptr[index] = dat.second->impl_get();
      this->h_particle_dat_ncomp_int.ptr[index] = dat.second->ncomp;
      index++;
    }

    // copy to the device
    EventStack event_stack;
    if (this->h_particle_dat_ptr_real.size_bytes() > 0) {
      event_stack.push(this->sycl_target->queue.memcpy(
          this->d_particle_dat_ptr_real.ptr, this->h_particle_dat_ptr_real.ptr,
          this->h_particle_dat_ptr_real.size_bytes()));
    }
    if (this->h_particle_dat_ptr_int.size_bytes() > 0) {
      event_stack.push(this->sycl_target->queue.memcpy(
          this->d_particle_dat_ptr_int.ptr, this->h_particle_dat_ptr_int.ptr,
          this->h_particle_dat_ptr_int.size_bytes()));
    }
    if (this->h_particle_dat_ncomp_real.size_bytes() > 0) {
      event_stack.push(this->sycl_target->queue.memcpy(
          this->d_particle_dat_ncomp_real.ptr,
          this->h_particle_dat_ncomp_real.ptr,
          this->h_particle_dat_ncomp_real.size_bytes()));
    }
    if (this->h_particle_dat_ncomp_int.size_bytes() > 0) {
      event_stack.push(this->sycl_target->queue.memcpy(
          this->d_particle_dat_ncomp_int.ptr,
          this->h_particle_dat_ncomp_int.ptr,
          this->h_particle_dat_ncomp_int.size_bytes()));
    }
    event_stack.wait();
  }

public:
  /// Disable (implicit) copies.
  CellMove(const CellMove &st) = delete;
  /// Disable (implicit) copies.
  CellMove &operator=(CellMove const &a) = delete;

  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;

  ~CellMove() {}
  /**
   * Create a cell move instance to move particles between cells.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param ncell Total number of cells.
   * @param layer_compressor LayerCompressor to use to compress ParticleDat
   * instances.
   * @param particle_dats_real Container of REAL ParticleDat.
   * @param particle_dats_int Container of INT ParticleDat.
   */
  CellMove(SYCLTargetSharedPtr sycl_target, const int ncell,
           LayerCompressor &layer_compressor,
           std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
           std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int)
      : ncell(ncell), sycl_target(sycl_target),
        layer_compressor(layer_compressor),
        particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int),
        h_npart_cell(sycl_target, this->ncell),
        d_npart_cell(sycl_target, this->ncell),
        d_cells_old(sycl_target, this->ncell),
        d_cells_new(sycl_target, this->ncell),
        d_layers_old(sycl_target, this->ncell),
        d_layers_new(sycl_target, this->ncell), h_move_count(sycl_target, 1),
        d_move_count(sycl_target, 1), h_particle_dat_ptr_real(sycl_target, 1),
        h_particle_dat_ptr_int(sycl_target, 1),
        h_particle_dat_ncomp_real(sycl_target, 1),
        h_particle_dat_ncomp_int(sycl_target, 1),
        d_particle_dat_ptr_real(sycl_target, 1),
        d_particle_dat_ptr_int(sycl_target, 1),
        d_particle_dat_ncomp_real(sycl_target, 1),
        d_particle_dat_ncomp_int(sycl_target, 1),
        ep_bad_cell_indices(sycl_target) {}

  /**
   * Set the ParticleDat to use as a source for cell ids.
   *
   * @param cell_id_dat ParticleDat to use for cell ids.
   */
  inline void set_cell_id_dat(ParticleDatSharedPtr<INT> cell_id_dat) {
    this->cell_id_dat = cell_id_dat;
  }

  /**
   * Move particles between cells (on this MPI rank) using the cell ids on
   * the particles.
   */
  inline void move();
};

} // namespace NESO::Particles

#endif
