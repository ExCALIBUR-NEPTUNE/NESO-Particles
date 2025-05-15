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

  BufferHost<int> h_npart_cell;
  BufferDevice<int> d_npart_cell;

  // references to the ParticleGroup methods
  ParticleDatSharedPtr<INT> cell_id_dat;
  std::map<Sym<REAL>, ParticleDatSharedPtr<REAL>> &particle_dats_real;
  std::map<Sym<INT>, ParticleDatSharedPtr<INT>> &particle_dats_int;

  EventStack event_stack;
  std::shared_ptr<ParticleLoopImplementation::ParticleLoopBlockIterationSet>
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
        h_npart_cell(sycl_target, ncell), d_npart_cell(sycl_target, ncell),
        particle_dats_real(particle_dats_real),
        particle_dats_int(particle_dats_int) {
    this->iteration_set = std::make_shared<
        ParticleLoopImplementation::ParticleLoopBlockIterationSet>(
        this->sycl_target, this->ncell, this->h_npart_cell.ptr,
        this->d_npart_cell.ptr);
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

#ifndef NESO_PARTICLES_TEST_COMPILATION
protected:
#endif

  template <typename T>
  inline void compute_remove_compress_indicies_dense(
      const int npart, T *usm_cells, T *usm_layers, int *compress_npart,
      INT *k_compress_cells_old, INT *k_compress_layers_old,
      INT *k_compress_layers_new, int *k_npart_cell_new) {
    if (npart < 1) {
      return;
    }

    auto r = ProfileRegion("LayerCompressor",
                           "compute_remove_compress_indicies_dense_a");

    int *k_npart_cell_dat_old =
        this->particle_dats_int[Sym<INT>("NESO_MPI_RANK")]->d_npart_cell;

    auto d_npart_cell_es_old =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_npart_cell_es_old->realloc_no_copy(this->ncell + 1);
    int *k_npart_cell_p1_old_es = d_npart_cell_es_old->ptr;
    auto e4 = this->sycl_target->queue.parallel_for(
        sycl::range<1>(ncell + 1),
        [=](auto ix) { k_npart_cell_p1_old_es[ix] = 0; });

    auto d_npart_cell_old =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_npart_cell_old->realloc_no_copy(this->ncell + 1);
    int *k_npart_cell_p1_old = d_npart_cell_old->ptr;

    // We collect the npart + 1 of each cell into a buffer that is ncell+1
    const int zero = 0;
    auto e0 = this->sycl_target->queue.memcpy(k_npart_cell_p1_old + this->ncell,
                                              &zero, sizeof(int));

    // The particle loop iteration set class reads this buffer to determine
    // the iteration set for a particle loop. Required on host not device.
    auto e1 = this->sycl_target->queue.memcpy(
        this->h_npart_cell.ptr, k_npart_cell_dat_old, ncell * sizeof(int));

    // Collect the actual npart cell (i.e. not with the +1) as well.
    int *k_npart_cell_old = this->d_npart_cell.ptr;
    this->sycl_target->queue
        .parallel_for(sycl::range<1>(this->ncell),
                      [=](auto idx) {
                        k_npart_cell_p1_old[idx] =
                            k_npart_cell_dat_old[idx] + 1;
                        k_npart_cell_old[idx] = k_npart_cell_dat_old[idx];
                      })
        .wait_and_throw();
    e0.wait_and_throw();
    e1.wait_and_throw();

    // Compute the exclusive scan of the occupancies which we need to call
    // joint_exclusive_scan_n
    e4.wait_and_throw();
    joint_exclusive_scan(this->sycl_target,
                         static_cast<std::size_t>(this->ncell + 1),
                         k_npart_cell_p1_old, k_npart_cell_p1_old_es)
        .wait_and_throw();

    int total_num_particles_p1 = -1;
    this->sycl_target->queue
        .memcpy(&total_num_particles_p1, k_npart_cell_p1_old_es + this->ncell,
                sizeof(int))
        .wait_and_throw();

    INT ***cell_ids_ptr = this->cell_id_dat->impl_get();
    const auto k_ncell = this->ncell;
    auto e2 = this->sycl_target->queue.parallel_for(
        sycl::range<1>(static_cast<size_t>(npart)), [=](sycl::id<1> idx) {
          const auto cell = usm_cells[idx];
          const auto layer = usm_layers[idx];
          const bool valid_cell = (-1 < cell) && (cell < k_ncell);
          const bool valid_layer = -1 < layer;

          //// indicate this particle is removed by setting
          /// the / cell index to -1
          if (valid_cell && valid_layer) {
            cell_ids_ptr[cell][0][layer] = -1;
          }
        });

    // Create space to store the entry flags for each particle
    auto d_masks = get_resource<BufferDevice<int>,
                                ResourceStackInterfaceBufferDevice<int>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
        sycl_target);
    d_masks->realloc_no_copy(total_num_particles_p1);
    auto k_masks = d_masks->ptr;

    // Create space to store the exclusive sums of the entries
    auto d_masks_es = get_resource<BufferDevice<int>,
                                   ResourceStackInterfaceBufferDevice<int>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
        sycl_target);
    d_masks_es->realloc_no_copy(total_num_particles_p1);
    auto k_masks_es = d_masks_es->ptr;
    auto e6 = this->sycl_target->queue.parallel_for(
        sycl::range<1>(total_num_particles_p1),
        [=](auto ix) { k_masks_es[ix] = 0; });

    e2.wait_and_throw();

    r.end();
    this->sycl_target->profile_map.add_region(r);
    r = ProfileRegion("LayerCompressor",
                      "compute_remove_compress_indicies_dense_b");

    // We now store in the d_masks array a 0 if the particle remains and a 1 if
    // the particle is removed.
    //
    auto is = this->iteration_set->get_all_cells(
        this->sycl_target->parameters->template get<SizeTParameter>("LOOP_NBIN")
            ->value);
    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(this->sycl_target->queue.parallel_for(
          blockx.loop_iteration_set, [=](auto idx) {
            std::size_t cell;
            std::size_t layer;
            block_device.get_cell_layer(idx, &cell, &layer);
            if (block_device.work_item_required(cell, layer)) {
              const int cell_offset = k_npart_cell_p1_old_es[cell];
              const int index = cell_offset + static_cast<int>(layer);
              k_masks[index] = (cell_ids_ptr[cell][0][layer] > -1) ? 0 : 1;
            }
          }));
    }
    this->event_stack.wait();

    r.end();
    this->sycl_target->profile_map.add_region(r);
    r = ProfileRegion("LayerCompressor",
                      "compute_remove_compress_indicies_dense_c");

    // Compute the exclusive sum cell wise for all of the masks.
    e6.wait_and_throw();
    joint_exclusive_scan_n(sycl_target, static_cast<std::size_t>(this->ncell),
                           k_npart_cell_p1_old, k_npart_cell_p1_old_es, k_masks,
                           k_masks_es)
        .wait_and_throw();

    r.end();
    this->sycl_target->profile_map.add_region(r);
    r = ProfileRegion("LayerCompressor",
                      "compute_remove_compress_indicies_dense_d");

    auto d_npart_to_fill =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_npart_to_fill->realloc_no_copy(this->ncell + 1);
    int *k_npart_to_fill = d_npart_to_fill->ptr;

    auto d_npart_to_fill_es =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_npart_to_fill_es->realloc_no_copy(this->ncell + 1);
    int *k_npart_to_fill_es = d_npart_to_fill_es->ptr;
    auto e5 = this->sycl_target->queue.parallel_for(
        sycl::range<1>(ncell + 1),
        [=](auto ix) { k_npart_to_fill_es[ix] = 0; });

    // Collect the new cell occupancies for each cell.
    this->sycl_target->queue
        .parallel_for(sycl::range<1>(this->ncell),
                      [=](auto cell) {
                        const int cell_offset = k_npart_cell_p1_old_es[cell];
                        const int last_index = k_npart_cell_p1_old[cell] - 1;
                        const int num_particles_removed =
                            k_masks_es[cell_offset + last_index];
                        const int new_cell_npart =
                            k_npart_cell_old[cell] - num_particles_removed;
                        k_npart_cell_new[cell] = new_cell_npart;
                        k_npart_to_fill[cell] =
                            k_masks_es[cell_offset + new_cell_npart];
                      })
        .wait_and_throw();

    e5.wait_and_throw();
    joint_exclusive_scan(this->sycl_target,
                         static_cast<std::size_t>(this->ncell + 1),
                         k_npart_to_fill, k_npart_to_fill_es)
        .wait_and_throw();

    auto e3 = this->sycl_target->queue.memcpy(
        compress_npart, k_npart_to_fill_es + this->ncell, sizeof(int));

    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(this->sycl_target->queue.parallel_for(
          blockx.loop_iteration_set, [=](auto idx) {
            std::size_t cell;
            std::size_t layer;
            block_device.get_cell_layer(idx, &cell, &layer);
            if (block_device.work_item_required(cell, layer)) {
              const int cell_offset_mask = k_npart_cell_p1_old_es[cell];
              const int index_mask = cell_offset_mask + static_cast<int>(layer);

              // if particle removed and before the new occupancy
              if (k_masks[index_mask] &&
                  (static_cast<int>(layer) < k_npart_cell_new[cell])) {
                const int index_output =
                    k_npart_to_fill_es[cell] + k_masks_es[index_mask];
                k_compress_cells_old[index_output] = static_cast<INT>(cell);
                k_compress_layers_new[index_output] = static_cast<INT>(layer);
              }
              // if particle NOT removed and after the new occupancy
              else if ((!k_masks[index_mask]) &&
                       (static_cast<int>(layer) >= k_npart_cell_new[cell])) {
                const int num_empty_before_index =
                    k_masks_es[index_mask] - k_npart_to_fill[cell];
                const int index_output =
                    k_npart_to_fill_es[cell] + static_cast<int>(layer) -
                    k_npart_cell_new[cell] - num_empty_before_index;
                k_compress_layers_old[index_output] = static_cast<INT>(layer);
              }
            }
          }));
    }
    this->event_stack.wait();
    e3.wait_and_throw();

    r.end();
    this->sycl_target->profile_map.add_region(r);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_npart_to_fill_es);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_npart_to_fill);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_masks_es);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_masks);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_npart_cell_old);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_npart_cell_es_old);
  }

  /**
   *  For the specified N particles to remove at the given cells and rows
   *  (layers) compute the data migration required to keep the particle data
   *  contiguous.
   *
   *  @param npart Number of particles, N, which are to be removed.
   *  @param usm_cells Device accessible pointers to an array of length N that
   * holds the cells of the particles that are to be removed.
   *  @param usm_layers Device accessible pointers to an array of length N that
   * holds the layers of the particles that are to be removed.
   */
  template <typename T>
  inline void compute_remove_compress_indicies_sparse(
      const int npart, T *usm_cells, T *usm_layers, int *compress_npart,
      INT *k_compress_cells_old, INT *k_compress_layers_old,
      INT *k_compress_layers_new, int *k_npart_cell_new) {

    if (npart < 1) {
      return;
    }

    auto t0 = profile_timestamp();
    auto r =
        ProfileRegion("LayerCompressor", "compute_remove_compress_indicies_a");

    const int ncell = this->ncell;

    auto mpi_particle_dat = this->particle_dats_int[Sym<INT>("NESO_MPI_RANK")];
    auto k_npart_cell_old = mpi_particle_dat->d_npart_cell;

    auto d_move_counters =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_move_counters->realloc_no_copy(ncell + 1);
    auto k_move_counters = d_move_counters->ptr;

    auto d_move_counters_es =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_move_counters_es->realloc_no_copy(ncell + 1);
    auto k_move_counters_es = d_move_counters_es->ptr;
    auto e4 = this->sycl_target->queue.parallel_for(
        sycl::range<1>(ncell + 1),
        [=](auto ix) { k_move_counters_es[ix] = 0; });

    auto d_compress_offsets =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_compress_offsets->realloc_no_copy(npart);
    auto k_compress_offsets = d_compress_offsets->ptr;

    INT ***cell_ids_ptr = this->cell_id_dat->impl_get();

    // Copy the current cell occupancies into k_npart_cell_new.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(ncell)), [=](sycl::id<1> idx) {
                k_npart_cell_new[idx] = static_cast<int>(k_npart_cell_old[idx]);
              });
        })
        .wait_and_throw();

    // Compute the new cell occupancies by atomically decrementing
    // k_npart_cell_new and setting the cell id of removed particles to
    // -1.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(static_cast<size_t>(npart)), [=](sycl::id<1> idx) {
                const auto cell = usm_cells[idx];
                const auto layer = usm_layers[idx];

                // Atomically do k_npart_cell_new[cell]--
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    element_atomic(k_npart_cell_new[cell]);
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
                [=](sycl::id<1> idx) { k_move_counters[idx] = 0; });
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
                if (layer < k_npart_cell_new[cell]) {
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      element_atomic(k_move_counters[cell]);
                  const int offset = element_atomic.fetch_add(1);
                  // We store this offset to avoid doing the same atomics later.
                  k_compress_offsets[idx] = offset;
                }
              });
        })
        .wait_and_throw();

    // We need to allocate space to find these particles but want to allocate
    // space sensibly in the case where the particle distribution is
    // non-uniform.
    e4.wait_and_throw();
    joint_exclusive_scan(this->sycl_target, this->ncell + 1, k_move_counters,
                         k_move_counters_es)
        .wait_and_throw();
    // We made to_find_scan have ncell+1 elements such that the final entry is
    // the number of particles we need to find across all cells.
    this->sycl_target->queue
        .memcpy(compress_npart, k_move_counters_es + this->ncell, sizeof(int))
        .wait_and_throw();

    NESOASSERT((0 <= *compress_npart) && (*compress_npart <= npart),
               "Bad compress_npart computed.");

    // Allocate the temporary space to be large enough to hold all the layer
    // indices.
    auto d_search_space = get_resource<BufferDevice<int>,
                                       ResourceStackInterfaceBufferDevice<int>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
        sycl_target);
    d_search_space->realloc_no_copy(*compress_npart);
    auto k_search_space = d_search_space->ptr;

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
                if (layer < k_npart_cell_new[cell]) {
                  const auto offset = k_compress_offsets[idx];
                  const auto index = k_move_counters_es[cell] + offset;
                  k_compress_cells_old[index] = cell;
                  k_compress_layers_old[index] = offset;
                  k_compress_layers_new[index] = layer;
                }
              });
        })
        .wait_and_throw();

    // Compute the iteration set sizes for each cell. This loop overrides the
    // k_move_counters array. This is the difference between the old
    // cell count and new cell count.
    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(static_cast<size_t>(ncell)),
                             [=](sycl::id<1> idx) {
                               k_move_counters[idx] = k_npart_cell_old[idx] -
                                                      k_npart_cell_new[idx];
                             });
        })
        .wait_and_throw();

    // The particle loop iteration set class reads this buffer to determine
    // the iteration set for a particle loop. Required on host not device.
    auto e0 = this->sycl_target->queue.memcpy(
        this->h_npart_cell.ptr, k_move_counters, ncell * sizeof(int));
    int *k_npart_cell = this->d_npart_cell.ptr;
    this->sycl_target->queue
        .parallel_for(sycl::range<1>(this->ncell),
                      [=](auto ix) { k_npart_cell[ix] = k_move_counters[ix]; })
        .wait_and_throw();
    e0.wait_and_throw();

    auto is = this->iteration_set->get_all_cells(
        this->sycl_target->parameters->template get<SizeTParameter>("LOOP_NBIN")
            ->value);
    // As the entries in k_move_counters were already copied to the
    // host to create the particle loop iteration sets we can zero the device
    // side buffer.
    lambda_reset_counters();

    for (auto &blockx : is) {
      const auto block_device = blockx.block_device;
      this->event_stack.push(this->sycl_target->queue.parallel_for(
          blockx.loop_iteration_set, [=](auto idx) {
            std::size_t cellx;
            std::size_t layerxs;
            block_device.get_cell_layer(idx, &cellx, &layerxs);
            // This takes the layer from the iteration set and offsets it
            // into the interval [new_cell_count, old_cell_count).
            const int layerx =
                static_cast<int>(layerxs) + k_npart_cell_new[cellx];
            // k_npart_cell_old still holds the old cell count.
            if (layerx < k_npart_cell_old[cellx]) {
              // Is this a particle to be copied lower in the cell?
              if (cell_ids_ptr[cellx][0][layerx] > -1) {
                // Place an ordering on the found particles to move for this
                // cell.
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    element_atomic(k_move_counters[cellx]);
                const int offset = element_atomic.fetch_add(1);

                // Store the index so we can retrive it later.
                // k_move_counters_es[cellx] holds the base offset for the
                // cell.
                const int store_index = k_move_counters_es[cellx] + offset;
                k_search_space[store_index] = layerx;
              }
            }
          }));
    }

    this->event_stack.wait();

    if ((*compress_npart) > 0) {
      this->sycl_target->queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for<>(
                sycl::range<1>(static_cast<size_t>(*compress_npart)),
                [=](sycl::id<1> idx) {
                  const auto cell = k_compress_cells_old[idx];
                  const auto offset = k_compress_layers_old[idx];
                  const auto cell_offset = k_move_counters_es[cell];
                  const auto lookup_index = cell_offset + offset;
                  const auto source_row = k_search_space[lookup_index];
                  k_compress_layers_old[idx] = source_row;
                });
          })
          .wait_and_throw();
    }

    sycl_target->profile_map.inc("LayerCompressor",
                                 "compute_remove_compress_indicies", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    r.end();
    this->sycl_target->profile_map.add_region(r);

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_search_space);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_move_counters);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_move_counters_es);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_compress_offsets);
  }

  bool test_mode{false};
  std::function<void(int compress_npart, INT *k_compress_cells_old,
                     INT *k_compress_layers_old, INT *k_compress_layers_new,
                     int *k_npart_cell_new)>
      callback_sparse;
  std::function<void(int compress_npart, INT *k_compress_cells_old,
                     INT *k_compress_layers_old, INT *k_compress_layers_new,
                     int *k_npart_cell_new)>
      callback_dense;
  std::function<void(const int npart, int *h_cells, int *h_layers,
                     int *h_npart_cell)>
      callback_host;
  std::function<void()> callback_test;

  template <typename T>
  inline void remove_particles_inner(const int npart, T *usm_cells,
                                     T *usm_layers) {

    // If there are no particles to remove then there is nothing to do.
    if (npart < 1) {
      return;
    }

    auto t0 = profile_timestamp();

    auto d_compress_cells_old =
        get_resource<BufferDevice<INT>,
                     ResourceStackInterfaceBufferDevice<INT>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<INT>{}, sycl_target);
    auto d_compress_layers_old =
        get_resource<BufferDevice<INT>,
                     ResourceStackInterfaceBufferDevice<INT>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<INT>{}, sycl_target);
    auto d_compress_layers_new =
        get_resource<BufferDevice<INT>,
                     ResourceStackInterfaceBufferDevice<INT>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<INT>{}, sycl_target);

    d_compress_cells_old->realloc_no_copy(npart);
    d_compress_layers_old->realloc_no_copy(npart);
    d_compress_layers_new->realloc_no_copy(npart);
    auto k_compress_cells_old = d_compress_cells_old->ptr;
    auto k_compress_layers_old = d_compress_layers_old->ptr;
    auto k_compress_layers_new = d_compress_layers_new->ptr;

    auto d_npart_cell_new =
        get_resource<BufferDevice<int>,
                     ResourceStackInterfaceBufferDevice<int>>(
            sycl_target->resource_stack_map,
            ResourceStackKeyBufferDevice<int>{}, sycl_target);
    d_npart_cell_new->realloc_no_copy(this->ncell);
    auto k_npart_cell_new = d_npart_cell_new->ptr;

    int compress_npart = 0;

    if (test_mode) {

      std::vector<T> h_cellsT(npart);
      std::vector<T> h_layersT(npart);

      this->sycl_target->queue
          .memcpy(h_cellsT.data(), usm_cells, npart * sizeof(T))
          .wait_and_throw();
      this->sycl_target->queue
          .memcpy(h_layersT.data(), usm_layers, npart * sizeof(T))
          .wait_and_throw();

      std::vector<int> h_cells(npart);
      std::vector<int> h_layers(npart);
      for (int ix = 0; ix < npart; ix++) {
        h_cells[ix] = h_cellsT[ix];
        h_layers[ix] = h_layersT[ix];
      }

      this->callback_host(npart, h_cells.data(), h_layers.data(),
                          this->cell_id_dat->h_npart_cell);

      this->compute_remove_compress_indicies_dense(
          npart, usm_cells, usm_layers, &compress_npart, k_compress_cells_old,
          k_compress_layers_old, k_compress_layers_new, k_npart_cell_new);
      this->callback_dense(compress_npart, k_compress_cells_old,
                           k_compress_layers_old, k_compress_layers_new,
                           k_npart_cell_new);

      this->compute_remove_compress_indicies_sparse(
          npart, usm_cells, usm_layers, &compress_npart, k_compress_cells_old,
          k_compress_layers_old, k_compress_layers_new, k_npart_cell_new);
      this->callback_sparse(compress_npart, k_compress_cells_old,
                            k_compress_layers_old, k_compress_layers_new,
                            k_npart_cell_new);

      this->callback_test();
    }

    // note to refactorers that this call uses h_npart_cell
    this->compute_remove_compress_indicies_dense(
        npart, usm_cells, usm_layers, &compress_npart, k_compress_cells_old,
        k_compress_layers_old, k_compress_layers_new, k_npart_cell_new);

    auto t1 = profile_timestamp();

    // do this d->h copy once for all dats
    if (this->ncell > 0) {
      this->event_stack.push(this->sycl_target->queue.memcpy(
          this->h_npart_cell.ptr, k_npart_cell_new, this->ncell * sizeof(int)));
    }

    auto r = ProfileRegion("LayerCompressor", "data_movement");

    std::size_t num_bytes = 0;
    for (auto &dat : particle_dats_real) {
      this->event_stack.push(dat.second->copy_particle_data(
          compress_npart, k_compress_cells_old, k_compress_cells_old,
          k_compress_layers_old, k_compress_layers_new));
      this->event_stack.push(
          dat.second->set_npart_cells_device(k_npart_cell_new));
      num_bytes += sizeof(REAL) * dat.second->ncomp;
    }
    for (auto &dat : particle_dats_int) {
      this->event_stack.push(dat.second->copy_particle_data(
          compress_npart, k_compress_cells_old, k_compress_cells_old,
          k_compress_layers_old, k_compress_layers_new));
      this->event_stack.push(
          dat.second->set_npart_cells_device(k_npart_cell_new));
      num_bytes += sizeof(INT) * dat.second->ncomp;
    }

    r.num_bytes = num_bytes * compress_npart;
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

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_compress_cells_old);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{},
                     d_compress_layers_old);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{},
                     d_compress_layers_new);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_npart_cell_new);
  }

public:
  /**
   * Remove particles from the ParticleDat instances.
   *
   * @param npart Number of particles to remove.
   * @param usm_cells Device accessible pointer to the particle cells.
   * @param usm_layers Device accessible pointer to the particle rows (layers).
   */
  void remove_particles(const int npart, int *usm_cells, int *usm_layers);

  /**
   * Remove particles from the ParticleDat instances.
   *
   * @param npart Number of particles to remove.
   * @param usm_cells Device accessible pointer to the particle cells.
   * @param usm_layers Device accessible pointer to the particle rows (layers).
   */
  void remove_particles(const int npart, INT *usm_cells, INT *usm_layers);
};

} // namespace NESO::Particles

#endif
