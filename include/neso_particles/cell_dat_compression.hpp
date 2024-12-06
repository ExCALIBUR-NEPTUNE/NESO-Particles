#ifndef _NESO_PARTICLES_CELL_DAT_COMPRESSION
#define _NESO_PARTICLES_CELL_DAT_COMPRESSION

#include <cstdint>
#include <map>
#include <memory>
#include <mpi.h>
#include <string>

#include "cell_dat.hpp"
#include "compute_target.hpp"
#include "loop/particle_loop_iteration_set.hpp"
#include "particle_dat.hpp"
#include "particle_set.hpp"
#include "particle_spec.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  A LayerCompressor is a tool to remove rows from ParticleDat instances that
 *  correspond to particle data where the particle has been removed. Particles
 *  are removed when they move between cells and when they are transferred to
 *  other MPI ranks. This ensures that particle data is not duplicated and that
 *  the particle data is contiguous.
 */
class LayerCompressor {
private:
  SYCLTargetSharedPtr sycl_target;
  const int ncell;
  int compress_npart;
  BufferDevice<INT> d_compress_cells_old;
  BufferDevice<INT> d_compress_layers_old;
  BufferDevice<INT> d_compress_layers_new;
  BufferDevice<INT> d_compress_offsets;

  // these should be INT not int but hipsycl refused to do atomic refs on long
  // int
  BufferDevice<int> d_npart_cell;
  BufferHost<int> h_npart_cell;
  BufferDevice<int> d_move_counters;
  BufferDevice<int> d_search_space;
  BufferDevice<int> d_to_find_exscan;

  // references to the ParticleGroup methods
  ParticleDatSharedPtr<INT> cell_id_dat;
  std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real;
  std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int;

  EventStack event_stack;
  std::shared_ptr<ParticleLoopImplementation::ParticleLoopIterationSet>
      iteration_set;

public:
  /// Disable (implicit) copies.
  LayerCompressor(const LayerCompressor &st) = delete;
  /// Disable (implicit) copies.
  LayerCompressor &operator=(LayerCompressor const &a) = delete;

  ~LayerCompressor() {}

  /**
   *  Construct a new layer compressor to compress all ParticleDat instances
   *  that are contained within the passed containers.
   *
   *  @param sycl_target SYCLTargetSharedPtr to use as compute device.
   *  @param ncell Number of cells within each ParticleDat.
   *  @param particle_dats_real Container of ParticleDat instances of REAL type.
   *  @param particle_dats_int Container of ParticleDat instances of INT type.
   */
  LayerCompressor(
      SYCLTargetSharedPtr sycl_target, const int ncell,
      std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real,
      std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int)
      : sycl_target(sycl_target), ncell(ncell),
        d_compress_cells_old(sycl_target, 1),
        d_compress_layers_old(sycl_target, 1),
        d_compress_layers_new(sycl_target, 1),
        d_compress_offsets(sycl_target, 1), d_npart_cell(sycl_target, ncell),
        h_npart_cell(sycl_target, ncell),
        d_move_counters(sycl_target, ncell + 1),
        d_search_space(sycl_target, ncell),
        d_to_find_exscan(sycl_target, ncell + 1),
        particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int) {
    this->iteration_set =
        std::make_shared<ParticleLoopImplementation::ParticleLoopIterationSet>(
            32, this->ncell, this->h_npart_cell.ptr);
  }

  /**
   *  Set the ParticleDat which contains the cell ids of the particles. This is
   *  not a function which is expected to be called by a user.
   *
   *  @param cell_id_dat ParticleDat containing particle cell ids.
   */
  inline void set_cell_id_dat(ParticleDatSharedPtr<INT> cell_id_dat) {
    this->cell_id_dat = cell_id_dat;
  }

  /**
   *  For the specified N particles to remove at the given cells and rows
   *  (layers) compute the data migration required to keep the particle data
   *  contiguous.
   *
   *  @param npart Number of particles which are to be removed.
   *  @param usm_cells Device accessible pointers to an array of length N that
   * holds the cells of the particles that are to be removed.
   *  @param usm_layers Device accessible pointers to an array of length N that
   * holds the layers of the particles that are to be removed.
   */
  template <typename T>
  inline void compute_remove_compress_indicies(const int npart, T *usm_cells,
                                               T *usm_layers) {
    auto t0 = profile_timestamp();
    auto r =
        ProfileRegion("LayerCompressor", "compute_remove_compress_indicies_a");

    NESOASSERT(this->d_npart_cell.size >= static_cast<std::size_t>(this->ncell),
               "Bad device_npart_cell length");

    const int ncell = this->ncell;

    auto mpi_particle_dat = this->particle_dats_int[Sym<INT>("NESO_MPI_RANK")];
    auto k_npart_cell = mpi_particle_dat->d_npart_cell;

    auto device_npart_cell_ptr = this->d_npart_cell.ptr;
    auto device_move_counters_ptr = this->d_move_counters.ptr;

    this->d_compress_offsets.realloc_no_copy(npart);
    auto compress_offsets_ptr = this->d_compress_offsets.ptr;

    INT ***cell_ids_ptr = this->cell_id_dat->impl_get();

    // Copy the current cell occupancies into device_npart_cell_ptr.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(ncell)),
                             [=](sycl::id<1> idx) {
                               device_npart_cell_ptr[idx] =
                                   static_cast<int>(k_npart_cell[idx]);
                             });
        })
        .wait_and_throw();

    // Compute the new cell occupancies by atomically decrementing
    // device_npart_cell_ptr and setting the cell id of removed particles to
    // -1.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(npart)), [=](sycl::id<1> idx) {
                const auto cell = usm_cells[idx];
                const auto layer = usm_layers[idx];

                // Atomically do device_npart_cell_ptr[cell]--
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    element_atomic(device_npart_cell_ptr[cell]);
                element_atomic.fetch_add(-1);

                //// indicate this particle is removed by setting
                /// the / cell index to -1
                cell_ids_ptr[cell][0][layer] = -1;
              });
        })
        .wait();

    r.end();
    this->sycl_target->profile_map.add_region(r);
    r = ProfileRegion("LayerCompressor", "compute_remove_compress_indicies_b");

    // Helper lambda that resets a counter per cell to 0.
    auto lambda_reset_counters = [&]() {
      this->sycl_target->queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for<>(
                sycl::range<1>(static_cast<size_t>(ncell + 1)),
                [=](sycl::id<1> idx) { device_move_counters_ptr[idx] = 0; });
          })
          .wait_and_throw();
    };
    lambda_reset_counters();

    // Determine how many particles need to be found per cell where these
    // particles are not to be removed but they have a layer greater than the
    // new cell occupancy.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(npart)), [=](sycl::id<1> idx) {
                const auto cell = usm_cells[idx];
                const auto layer = usm_layers[idx];
                // Is this layer less than the new cell count?
                // If so then there is a particle in a row greater than the cell
                // count to be copied into this layer.
                if (layer < device_npart_cell_ptr[cell]) {
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      element_atomic(device_move_counters_ptr[cell]);
                  const int offset = element_atomic.fetch_add(1);
                  // We store this offset to avoid doing the same atomics later.
                  compress_offsets_ptr[idx] = offset;
                }
              });
        })
        .wait_and_throw();

    // We need to allocate space to find these particles but want to allocate
    // space sensibly in the case where the particle distribution is
    // non-uniform.
    auto to_find_scan_ptr = this->d_to_find_exscan.ptr;
    joint_exclusive_scan(this->sycl_target, this->ncell + 1,
                         device_move_counters_ptr, to_find_scan_ptr)
        .wait_and_throw();
    // We made to_find_scan have ncell+1 elements such that the final entry is
    // the number of particles we need to find across all cells.
    this->sycl_target->queue
        .memcpy(&this->compress_npart, to_find_scan_ptr + this->ncell,
                sizeof(int))
        .wait_and_throw();
    // Allocate the temporary space to be large enough to hold all the layer
    // indices.
    this->d_search_space.realloc_no_copy(this->compress_npart);

    this->d_compress_cells_old.realloc_no_copy(this->compress_npart);
    this->d_compress_layers_old.realloc_no_copy(this->compress_npart);
    this->d_compress_layers_new.realloc_no_copy(this->compress_npart);
    auto compress_cells_old_ptr = this->d_compress_cells_old.ptr;
    auto compress_layers_old_ptr = this->d_compress_layers_old.ptr;
    auto compress_layers_new_ptr = this->d_compress_layers_new.ptr;

    // Remove the indices which we do not need any more.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(npart)), [=](sycl::id<1> idx) {
                const auto cell = usm_cells[idx];
                const auto layer = usm_layers[idx];
                // Is this layer less than the new cell count?
                // If so then there is a particle in a row greater than the cell
                // count to be copied into this layer.
                if (layer < device_npart_cell_ptr[cell]) {
                  const auto offset = compress_offsets_ptr[idx];
                  const auto index = to_find_scan_ptr[cell] + offset;
                  compress_cells_old_ptr[index] = cell;
                  compress_layers_old_ptr[index] = offset;
                  compress_layers_new_ptr[index] = layer;
                }
              });
        })
        .wait_and_throw();

    // Compute the iteration set sizes for each cell. This loop overrides the
    // device_move_counters_ptr array. This is the difference between the old
    // cell count and new cell count.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(ncell)), [=](sycl::id<1> idx) {
                device_move_counters_ptr[idx] =
                    k_npart_cell[idx] - device_npart_cell_ptr[idx];
              });
        })
        .wait_and_throw();

    // The particle loop iteration set class reads this buffer to determine
    // the iteration set for a particle loop. Required on host not device.
    this->sycl_target->queue
        .memcpy(this->h_npart_cell.ptr, device_move_counters_ptr,
                ncell * sizeof(int))
        .wait_and_throw();

    auto is = this->iteration_set->get();
    // As the entries in device_move_counters_ptr were already copied to the
    // host to create the particle loop iteration sets we can zero the device
    // side buffer.
    lambda_reset_counters();
    auto k_search_space = this->d_search_space.ptr;

    const int nbin = std::get<0>(is);
    for (int binx = 0; binx < nbin; binx++) {
      sycl::nd_range<2> ndr = std::get<1>(is).at(binx);
      const size_t cell_offset = std::get<2>(is).at(binx);
      this->event_stack.push(
          this->sycl_target->queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<>(ndr, [=](sycl::nd_item<2> idx) {
              const size_t cellxs = idx.get_global_id(0) + cell_offset;
              const size_t layerxs = idx.get_global_id(1);
              const int cellx = static_cast<int>(cellxs);
              // This takes the layer from the iteration set and offsets it
              // into the interval [new_cell_count, old_cell_count).
              const int layerx =
                  static_cast<int>(layerxs) + device_npart_cell_ptr[cellxs];
              // k_npart_cell still holds the old cell count.
              if (layerx < k_npart_cell[cellx]) {
                // Is this a particle to be copied lower in the cell?
                if (cell_ids_ptr[cellx][0][layerx] > -1) {
                  // Place an ordering on the found particles to move for this
                  // cell.
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      element_atomic(device_move_counters_ptr[cellx]);
                  const int offset = element_atomic.fetch_add(1);

                  // Store the index so we can retrive it later.
                  // to_find_scan_ptr[cellx] holds the base offset for the cell.
                  const int store_index = to_find_scan_ptr[cellx] + offset;
                  k_search_space[store_index] = layerx;
                }
              }
            });
          }));
    }
    this->event_stack.wait();

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(this->compress_npart)),
              [=](sycl::id<1> idx) {
                const auto cell = compress_cells_old_ptr[idx];
                const auto offset = compress_layers_old_ptr[idx];
                const auto cell_offset = to_find_scan_ptr[cell];
                const auto lookup_index = cell_offset + offset;
                const auto source_row = k_search_space[lookup_index];
                compress_layers_old_ptr[idx] = source_row;
              });
        })
        .wait_and_throw();

    sycl_target->profile_map.inc("LayerCompressor",
                                 "compute_remove_compress_indicies", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    r.end();
    this->sycl_target->profile_map.add_region(r);
  }

  /**
   * Remove particles from the ParticleDat instances.
   *
   * @param npart Number of particles to remove.
   * @param usm_cells Device accessible pointer to the particle cells.
   * @param usm_layers Device accessible pointer to the particle rows (layers).
   */
  template <typename T>
  inline void remove_particles(const int npart, T *usm_cells, T *usm_layers) {

    // If there are no particles to remove then there is nothing to do.
    if (npart < 1) {
      return;
    }

    auto t0 = profile_timestamp();

    // note to refactorers that this call uses h_npart_cell
    this->compute_remove_compress_indicies(npart, usm_cells, usm_layers);

    auto compress_cells_old_ptr = this->d_compress_cells_old.ptr;
    auto compress_layers_old_ptr = this->d_compress_layers_old.ptr;
    auto compress_layers_new_ptr = this->d_compress_layers_new.ptr;

    auto t1 = profile_timestamp();

    // do this d->h copy once for all dats
    if (this->d_npart_cell.size_bytes() > 0) {
      this->event_stack.push(this->sycl_target->queue.memcpy(
          this->h_npart_cell.ptr, this->d_npart_cell.ptr,
          this->d_npart_cell.size_bytes()));
    }

    auto r = ProfileRegion("LayerCompressor", "data_movement");

    std::size_t num_bytes = 0;
    for (auto &dat : particle_dats_real) {
      this->event_stack.push(dat.second->copy_particle_data(
          this->compress_npart, compress_cells_old_ptr, compress_cells_old_ptr,
          compress_layers_old_ptr, compress_layers_new_ptr));
      this->event_stack.push(
          dat.second->set_npart_cells_device(this->d_npart_cell.ptr));
      num_bytes += sizeof(REAL) * dat.second->ncomp;
    }
    for (auto &dat : particle_dats_int) {
      this->event_stack.push(dat.second->copy_particle_data(
          this->compress_npart, compress_cells_old_ptr, compress_cells_old_ptr,
          compress_layers_old_ptr, compress_layers_new_ptr));
      this->event_stack.push(
          dat.second->set_npart_cells_device(this->d_npart_cell.ptr));
      num_bytes += sizeof(INT) * dat.second->ncomp;
    }

    r.num_bytes = num_bytes * this->compress_npart;
    // the move and set_npart calls are async
    this->event_stack.wait();

    r.end();
    this->sycl_target->profile_map.add_region(r);
    sycl_target->profile_map.inc("LayerCompressor", "data_movement", 1,
                                 profile_elapsed(t1, profile_timestamp()));

    r = ProfileRegion("LayerCompressor", "dat_bookkeeping");

    auto t2 = profile_timestamp();
    for (auto &dat : particle_dats_real) {
      dat.second->set_npart_cells_host(this->h_npart_cell.ptr);
    }
    for (auto &dat : particle_dats_int) {
      dat.second->set_npart_cells_host(this->h_npart_cell.ptr);
    }
    sycl_target->profile_map.inc("LayerCompressor", "host_npart_setting", 1,
                                 profile_elapsed(t2, profile_timestamp()));

    auto t3 = profile_timestamp();
    for (auto &dat : particle_dats_real) {
      dat.second->trim_cell_dat_rows();
    }
    for (auto &dat : particle_dats_int) {
      dat.second->trim_cell_dat_rows();
    }
    sycl_target->profile_map.inc("LayerCompressor", "dat_trimming", 1,
                                 profile_elapsed(t3, profile_timestamp()));

    auto t4 = profile_timestamp();
    for (auto &dat : particle_dats_real) {
      dat.second->cell_dat.wait_set_nrow();
    }
    for (auto &dat : particle_dats_int) {
      dat.second->cell_dat.wait_set_nrow();
    }

    sycl_target->profile_map.inc("LayerCompressor", "dat_trimming_wait", 1,
                                 profile_elapsed(t4, profile_timestamp()));

    sycl_target->profile_map.inc("LayerCompressor", "remove_particles", 1,
                                 profile_elapsed(t0, profile_timestamp()));

    r.end();
    this->sycl_target->profile_map.add_region(r);
  }
};

} // namespace NESO::Particles

#endif
