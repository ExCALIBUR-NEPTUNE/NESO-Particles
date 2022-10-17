#ifndef _NESO_PARTICLES_CELL_DAT_MOVE
#define _NESO_PARTICLES_CELL_DAT_MOVE

#include <CL/sycl.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "cell_dat.hpp"
#include "compute_target.hpp"
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

  ParticleDatShPtr<INT> cell_id_dat;
  std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real;
  std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int;

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
      this->h_particle_dat_ptr_real.ptr[index] =
          dat.second->cell_dat.device_ptr();
      this->h_particle_dat_ncomp_real.ptr[index] = dat.second->ncomp;
      index++;
    }
    index = 0;
    for (auto &dat : particle_dats_int) {
      this->h_particle_dat_ptr_int.ptr[index] =
          dat.second->cell_dat.device_ptr();
      this->h_particle_dat_ncomp_int.ptr[index] = dat.second->ncomp;
      index++;
    }

    // copy to the device
    EventStack event_stack;
    if (this->h_particle_dat_ptr_real.size_bytes() > 0) {
      event_stack.push(this->sycl_target.queue.memcpy(
          this->d_particle_dat_ptr_real.ptr, this->h_particle_dat_ptr_real.ptr,
          this->h_particle_dat_ptr_real.size_bytes()));
    }
    if (this->h_particle_dat_ptr_int.size_bytes() > 0) {
      event_stack.push(this->sycl_target.queue.memcpy(
          this->d_particle_dat_ptr_int.ptr, this->h_particle_dat_ptr_int.ptr,
          this->h_particle_dat_ptr_int.size_bytes()));
    }
    if (this->h_particle_dat_ncomp_real.size_bytes() > 0) {
      event_stack.push(this->sycl_target.queue.memcpy(
          this->d_particle_dat_ncomp_real.ptr,
          this->h_particle_dat_ncomp_real.ptr,
          this->h_particle_dat_ncomp_real.size_bytes()));
    }
    if (this->h_particle_dat_ncomp_int.size_bytes() > 0) {
      event_stack.push(this->sycl_target.queue.memcpy(
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
  SYCLTarget &sycl_target;

  ~CellMove() {}
  /**
   * Create a cell move instance to move particles between cells.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param ncell Total number of cells.
   * @param layer_compressor LayerCompressor to use to compress ParticleDat
   * instances.
   * @param particle_dats_real Container of REAL ParticleDat.
   * @param particle_dats_int Container of INT ParticleDat.
   */
  CellMove(SYCLTarget &sycl_target, const int ncell,
           LayerCompressor &layer_compressor,
           std::map<Sym<REAL>, ParticleDatShPtr<REAL>> &particle_dats_real,
           std::map<Sym<INT>, ParticleDatShPtr<INT>> &particle_dats_int)
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
  inline void set_cell_id_dat(ParticleDatShPtr<INT> cell_id_dat) {
    this->cell_id_dat = cell_id_dat;
  }

  /**
   * Move particles between cells (on this MPI rank) using the cell ids on
   * the particles.
   */
  inline void move() {
    auto t0 = profile_timestamp();
    // reset the particle counters on each cell
    auto mpi_rank_dat = particle_dats_int[Sym<INT>("NESO_MPI_RANK")];
    const auto k_ncell = this->ncell;
    auto k_npart_cell = d_npart_cell.ptr;
    auto k_mpi_npart_cell = mpi_rank_dat->d_npart_cell;
    auto reset_event = this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<1>(k_ncell), [=](sycl::id<1> idx) {
        k_npart_cell[idx] = k_mpi_npart_cell[idx];
      });
    });
    auto k_move_count = d_move_count.ptr;
    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.single_task<>([=]() { k_move_count[0] = 0; });
        })
        .wait_and_throw();
    reset_event.wait_and_throw();

    // space to store particles moving between cells
    const INT npart_local = mpi_rank_dat->get_npart_local();
    this->d_cells_old.realloc_no_copy(npart_local);
    this->d_cells_new.realloc_no_copy(npart_local);
    this->d_layers_old.realloc_no_copy(npart_local);
    this->d_layers_new.realloc_no_copy(npart_local);

    auto k_cells_old = this->d_cells_old.ptr;
    auto k_cells_new = this->d_cells_new.ptr;
    auto k_layers_old = this->d_layers_old.ptr;
    auto k_layers_new = this->d_layers_new.ptr;

    // loop over particles and identify the particles to be move between
    // cells.
    auto pl_iter_range = mpi_rank_dat->get_particle_loop_iter_range();
    auto pl_stride = mpi_rank_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = mpi_rank_dat->get_particle_loop_npart_cell();
    auto k_cell_id_dat = this->cell_id_dat->cell_dat.device_ptr();

    // detect out of bounds particles
    auto k_ep_indices = this->ep_bad_cell_indices.device_ptr();

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                // if the cell on the particle is not the current cell then
                // the particle needs moving.
                const auto cell_on_dat = k_cell_id_dat[cellx][0][layerx];

                const bool valid_cell =
                    (cell_on_dat >= 0) && (cell_on_dat < k_ncell);
                NESO_KERNEL_ASSERT(valid_cell, k_ep_indices);

                if ((cellx != cell_on_dat) && valid_cell) {
                  // Atomically increment the particle count for the new
                  // cell
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      atomic_layer(k_npart_cell[cell_on_dat]);
                  const int layer_new = atomic_layer.fetch_add(1);

                  // Get an index for this particle in the arrays that hold
                  // old/new cells/layers
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      atomic_index(k_move_count[0]);
                  const int array_index = atomic_index.fetch_add(1);

                  k_cells_old[array_index] = cellx;
                  k_cells_new[array_index] = static_cast<int>(cell_on_dat);
                  k_layers_old[array_index] = layerx;
                  k_layers_new[array_index] = layer_new;
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    this->ep_bad_cell_indices.check_and_throw(
        "Particle held bad cell id (not in [0,..,N_cell - 1]).");

    // Realloc the ParticleDat cells for the move
    if (this->ncell > 0) {
      this->sycl_target.queue
          .memcpy(this->h_npart_cell.ptr, this->d_npart_cell.ptr,
                  sizeof(int) * this->ncell)
          .wait();
    }

    for (auto &dat : particle_dats_real) {
      dat.second->realloc(this->h_npart_cell);
    }
    for (auto &dat : particle_dats_int) {
      dat.second->realloc(this->h_npart_cell);
    }

    // wait for the reallocs
    for (auto &dat : particle_dats_real) {
      dat.second->wait_realloc();
    }
    for (auto &dat : particle_dats_int) {
      dat.second->wait_realloc();
    }
    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<class dummy>(
              sycl::range<1>(k_ncell),
              [=](sycl::id<1> idx) { k_npart_cell[idx] = 0; });
        })
        .wait();

    EventStack tmp_stack;
    for (auto &dat : particle_dats_real) {
      tmp_stack.push(dat.second->async_set_npart_cells(this->h_npart_cell));
    }
    for (auto &dat : particle_dats_int) {
      tmp_stack.push(dat.second->async_set_npart_cells(this->h_npart_cell));
    }
    tmp_stack.wait();

    // get the npart to move on the host
    this->sycl_target.queue
        .memcpy(this->h_move_count.ptr, this->d_move_count.ptr, sizeof(int))
        .wait();
    const int move_count = h_move_count.ptr[0];

    // get the pointers into the ParticleDats
    this->get_particle_dat_info();

    const int k_num_dats_real = this->num_dats_real;
    const int k_num_dats_int = this->num_dats_int;
    const auto k_particle_dat_ptr_real = this->d_particle_dat_ptr_real.ptr;
    const auto k_particle_dat_ptr_int = this->d_particle_dat_ptr_int.ptr;
    const auto k_particle_dat_ncomp_real = this->d_particle_dat_ncomp_real.ptr;
    const auto k_particle_dat_ncomp_int = this->d_particle_dat_ncomp_int.ptr;

    auto t1 = profile_timestamp();
    // copy from old cells/layers to new cells/layers
    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(move_count), [=](sycl::id<1> idx) {
            const auto cell_old = k_cells_old[idx];
            const auto cell_new = k_cells_new[idx];
            const auto layer_old = k_layers_old[idx];
            const auto layer_new = k_layers_new[idx];

            // loop over the ParticleDats and copy the data
            // for each real dat
            for (int dx = 0; dx < k_num_dats_real; dx++) {
              REAL ***dat_ptr = k_particle_dat_ptr_real[dx];
              const int ncomp = k_particle_dat_ncomp_real[dx];
              // for each component
              for (int cx = 0; cx < ncomp; cx++) {
                dat_ptr[cell_new][cx][layer_new] =
                    dat_ptr[cell_old][cx][layer_old];
              }
            }
            // for each int dat
            for (int dx = 0; dx < k_num_dats_int; dx++) {
              INT ***dat_ptr = k_particle_dat_ptr_int[dx];
              const int ncomp = k_particle_dat_ncomp_int[dx];
              // for each component
              for (int cx = 0; cx < ncomp; cx++) {
                dat_ptr[cell_new][cx][layer_new] =
                    dat_ptr[cell_old][cx][layer_old];
              }
            }
          });
        })
        .wait_and_throw();
    sycl_target.profile_map.inc("CellMove", "cell_move", 1,
                                profile_elapsed(t1, profile_timestamp()));

    auto t2 = profile_timestamp();
    // compress the data by removing the old rows
    this->layer_compressor.remove_particles(move_count, this->d_cells_old.ptr,
                                            this->d_layers_old.ptr);

    sycl_target.profile_map.inc("CellMove", "remove_particles", 1,
                                profile_elapsed(t2, profile_timestamp()));
    sycl_target.profile_map.inc("CellMove", "move", 1,
                                profile_elapsed(t0, profile_timestamp()));
  };
};

} // namespace NESO::Particles

#endif
