#ifndef _NESO_PARTICLES_GLOBAL_MAPPING_IMPL_H_
#define _NESO_PARTICLES_GLOBAL_MAPPING_IMPL_H_

#include "global_mapping.hpp"
#include "loop/particle_loop.hpp"

namespace NESO::Particles {

/**
 * For each particle that does not have a non-negative MPI rank determined as
 * a local owner obtain the MPI rank that owns the global cell which contains
 * the particle.
 */
inline void MeshHierarchyGlobalMap::execute() {
  auto t0 = profile_timestamp();

  // reset the device count for cell ids that need mapping
  auto k_lookup_count = this->d_lookup_count.ptr;
  auto reset_event = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
    cgh.single_task<>([=]() { k_lookup_count[0] = 0; });
  });

  const auto npart_local = this->mpi_rank_dat->get_npart_local();
  this->d_lookup_local_cells.realloc_no_copy(npart_local);
  this->d_lookup_local_layers.realloc_no_copy(npart_local);

  reset_event.wait();

  // pointers to access dats in kernel
  auto k_position_dat = this->position_dat->impl_get_const();
  auto k_mpi_rank_dat = this->mpi_rank_dat->impl_get();

  // pointers to access BufferDevices in the kernel
  auto k_lookup_local_cells = this->d_lookup_local_cells.ptr;
  auto k_lookup_local_layers = this->d_lookup_local_layers.ptr;

  // detect errors in the kernel
  auto k_error_propagate = this->error_propagate.device_ptr();
  const auto k_npart_local = npart_local;

  ParticleLoop(
      "global_map_stage_0", this->mpi_rank_dat,
      [=](auto loop_index, auto mpi_rank_dat) {
        const int cellx = loop_index.cell;
        const int layerx = loop_index.layer;
        // Inspect to see if the a mpi rank has been identified for a
        // local communication pattern.
        const auto mpi_rank_on_dat = mpi_rank_dat[1];
        if (mpi_rank_on_dat < 0) {
          // Atomically increment the lookup count
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              atomic_count(k_lookup_count[0]);
          const int index = atomic_count.fetch_add(1);

          NESO_KERNEL_ASSERT((index >= 0) && (index < k_npart_local),
                             k_error_propagate);

          // store this particles location so that it can be
          // directly accessed later
          k_lookup_local_cells[index] = cellx;
          k_lookup_local_layers[index] = layerx;
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(this->mpi_rank_dat))
      .execute();

  this->error_propagate.check_and_throw("Bad atomic index computed.");

  this->sycl_target->queue
      .memcpy(this->h_lookup_count.ptr, this->d_lookup_count.ptr,
              this->d_lookup_count.size_bytes())
      .wait();
  const auto npart_query = this->h_lookup_count.ptr[0];

  // these global indices are passed to the mesh hierarchy toget the rank
  this->d_lookup_global_cells.realloc_no_copy(npart_query * 6);
  this->h_lookup_global_cells.realloc_no_copy(npart_query * 6);
  this->h_lookup_ranks.realloc_no_copy(npart_query);
  this->d_lookup_ranks.realloc_no_copy(npart_query);

  auto k_lookup_global_cells = this->d_lookup_global_cells.ptr;
  auto k_lookup_ranks = this->d_lookup_ranks.ptr;

  // variables required in the kernel to map positions to cells
  auto mesh_hierarchy = this->h_mesh->get_mesh_hierarchy();
  auto k_ndim = mesh_hierarchy->ndim;
  const REAL k_inverse_cell_width_coarse =
      mesh_hierarchy->inverse_cell_width_coarse;
  const REAL k_inverse_cell_width_fine =
      mesh_hierarchy->inverse_cell_width_fine;
  const REAL k_cell_width_coarse = mesh_hierarchy->cell_width_coarse;
  const REAL k_cell_width_fine = mesh_hierarchy->cell_width_fine;
  const INT k_ncells_dim_fine = mesh_hierarchy->ncells_dim_fine;

  for (int dimx = 0; dimx < k_ndim; dimx++) {
    this->h_origin.ptr[dimx] = mesh_hierarchy->origin[dimx];
    this->h_dims.ptr[dimx] = mesh_hierarchy->dims[dimx];
  }

  this->sycl_target->queue
      .memcpy(this->d_origin.ptr, this->h_origin.ptr,
              this->h_origin.size_bytes())
      .wait();

  this->sycl_target->queue
      .memcpy(this->d_dims.ptr, this->h_dims.ptr, this->h_dims.size_bytes())
      .wait();

  auto k_origin = this->d_origin.ptr;
  auto k_dims = this->d_dims.ptr;

  // map particles positions to coarse and fine cells in the mesh hierarchy
  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(npart_query), [=](sycl::id<1> idx) {
          const INT cellx = k_lookup_local_cells[idx];
          const INT layerx = k_lookup_local_layers[idx];

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            // position relative to the mesh origin
            const REAL pos =
                k_position_dat[cellx][dimx][layerx] - k_origin[dimx];
            const REAL tol = 1.0e-10;

            // coarse grid index
            INT cell_coarse = ((REAL)pos * k_inverse_cell_width_coarse);
            // bounds check the cell at the upper extent
            if (cell_coarse >= k_dims[dimx]) {
              // if the particle is within a given tolerance assume the
              // out of bounds is a floating point issue.
              if ((ABS(pos - k_dims[dimx] * k_cell_width_coarse) / ABS(pos)) <=
                  tol) {
                cell_coarse = k_dims[dimx] - 1;
                k_lookup_global_cells[(idx * k_ndim * 2) + dimx] = cell_coarse;
              } else {
                cell_coarse = 0;
                k_lookup_global_cells[(idx * k_ndim * 2) + dimx] = -2;
                NESO_KERNEL_ASSERT(false, k_error_propagate);
              }
            } else {
              k_lookup_global_cells[(idx * k_ndim * 2) + dimx] = cell_coarse;
            }

            // use the coarse cell index to offset the origin and compute
            // the fine cell index
            const REAL pos_fine = pos - cell_coarse * k_cell_width_coarse;
            INT cell_fine = ((REAL)pos_fine * k_inverse_cell_width_fine);

            if (cell_fine >= k_ncells_dim_fine) {
              if ((ABS(pos_fine - k_ncells_dim_fine * k_cell_width_fine) /
                   ABS(pos_fine)) <= tol) {
                cell_fine = k_ncells_dim_fine - 1;
                k_lookup_global_cells[(idx * k_ndim * 2) + dimx + k_ndim] =
                    cell_fine;
              } else {
                k_lookup_global_cells[(idx * k_ndim * 2) + dimx + k_ndim] = -2;
                NESO_KERNEL_ASSERT(false, k_error_propagate);
              }
            } else {
              k_lookup_global_cells[(idx * k_ndim * 2) + dimx + k_ndim] =
                  cell_fine;
            }
          }
        });
      })
      .wait_and_throw();

  this->error_propagate.check_and_throw(
      "Could not bin into MeshHierarchy cell.");

  // copy the computed indicies to the host
  if (this->d_lookup_global_cells.size_bytes() > 0) {
    this->sycl_target->queue
        .memcpy(this->h_lookup_global_cells.ptr,
                this->d_lookup_global_cells.ptr,
                this->d_lookup_global_cells.size_bytes())
        .wait_and_throw();
  }
  // get the mpi ranks
  mesh_hierarchy->get_owners(npart_query, this->h_lookup_global_cells.ptr,
                             this->h_lookup_ranks.ptr);

  // copy the mpi ranks back to the ParticleDat
  if (this->h_lookup_ranks.size_bytes() > 0) {
    this->sycl_target->queue
        .memcpy(this->d_lookup_ranks.ptr, this->h_lookup_ranks.ptr,
                this->h_lookup_ranks.size_bytes())
        .wait_and_throw();
  }

  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(npart_query), [=](sycl::id<1> idx) {
          const INT cellx = k_lookup_local_cells[idx];
          const INT layerx = k_lookup_local_layers[idx];
          k_mpi_rank_dat[cellx][0][layerx] = k_lookup_ranks[idx];
        });
      })
      .wait_and_throw();

  sycl_target->profile_map.inc("MeshHierarchyGlobalMap", "execute", 1,
                               profile_elapsed(t0, profile_timestamp()));
}

/**
 *  Set all components 0 and 1 of particles to -1 in the passed ParticleDat.
 *
 *  @param mpi_rank_dat ParticleDat containing MPI ranks to reset.
 */
inline void reset_mpi_ranks(ParticleDatSharedPtr<INT> mpi_rank_dat) {
  ParticleLoop(
      "reset_mpi_ranks", mpi_rank_dat,
      [=](auto mpi_rank_dat) {
        mpi_rank_dat[0] = -1;
        mpi_rank_dat[1] = -1;
      },
      Access::write(mpi_rank_dat))
      .execute();
}

} // namespace NESO::Particles
#endif
