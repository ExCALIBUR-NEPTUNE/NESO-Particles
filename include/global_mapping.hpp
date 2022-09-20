#ifndef _NESO_PARTICLES_GLOBAL_MAPPING
#define _NESO_PARTICLES_GLOBAL_MAPPING

#include <CL/sycl.hpp>
#include <mpi.h>

#include "domain.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/**
 * This class maps global positions into the cells of a HMesh and determines
 * which MPI rank owns that cell.
 */
class MeshHierarchyGlobalMap {
private:
  /// Compute device used by the instance.
  SYCLTarget &sycl_target;
  /// HMesh instance on which particles live.
  HMesh &h_mesh;
  /// ParticleDat storing Positions
  ParticleDatShPtr<REAL> &position_dat;
  /// ParticleDat storing cell ids
  ParticleDatShPtr<INT> &cell_id_dat;
  /// ParticleDat storing MPI rank
  ParticleDatShPtr<INT> &mpi_rank_dat;

  /// Host buffer containing the number of particles to query owning MPI rank
  BufferHost<int> h_lookup_count;
  /// Device buffer containing the number of particles to query owning MPI rank
  BufferDevice<int> d_lookup_count;

  /// Host buffer of global cells on HMesh to query owner of.
  BufferHost<INT> h_lookup_global_cells;
  /// Device buffer of global cells on HMesh to query owner of.
  BufferDevice<INT> d_lookup_global_cells;
  /// Space to store the ranks owning the lookup cells on the host.
  BufferHost<int> h_lookup_ranks;
  /// Space to store the ranks owning the lookup cells on the Device.
  BufferDevice<int> d_lookup_ranks;

  /// Cells of the particles for which the lookup is being performed.
  BufferDevice<int> d_lookup_local_cells;
  /// Layers of the particles for which the lookup is being performed.
  BufferDevice<int> d_lookup_local_layers;

  /// Host buffer holding the origin of MeshHierarchy
  BufferHost<REAL> h_origin;
  /// Device buffer holding the origin of MeshHierarchy
  BufferDevice<REAL> d_origin;
  /// Host buffer containing the dims of MeshHierarchy
  BufferHost<int> h_dims;
  /// Device buffer containing the dims of MeshHierarchy
  BufferDevice<int> d_dims;

public:
  ~MeshHierarchyGlobalMap(){};

  /**
   * Construct a new global mapping instance for MeshHierarchy.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param h_mesh HMesh derived mesh to use for mapping.
   * @param position_dat ParticleDat containing particle positions.
   * @param cell_id_dat ParticleDat containg particle cell ids.
   * @param mpi_rank_dat ParticleDat containing the owning rank of particles.
   */
  MeshHierarchyGlobalMap(SYCLTarget &sycl_target, HMesh &h_mesh,
                         ParticleDatShPtr<REAL> &position_dat,
                         ParticleDatShPtr<INT> &cell_id_dat,
                         ParticleDatShPtr<INT> &mpi_rank_dat)
      : sycl_target(sycl_target), h_mesh(h_mesh), position_dat(position_dat),
        cell_id_dat(cell_id_dat), mpi_rank_dat(mpi_rank_dat),
        h_lookup_count(sycl_target, 1), d_lookup_count(sycl_target, 1),
        h_lookup_global_cells(sycl_target, 1),
        d_lookup_global_cells(sycl_target, 1), h_lookup_ranks(sycl_target, 1),
        d_lookup_ranks(sycl_target, 1), d_lookup_local_cells(sycl_target, 1),
        d_lookup_local_layers(sycl_target, 1), h_origin(sycl_target, 3),
        d_origin(sycl_target, 3), h_dims(sycl_target, 3),
        d_dims(sycl_target, 3){};

  /**
   * For each particle that does not have a non-negative MPI rank determined as
   * a local owner obtain the MPI rank that owns the global cell which contains
   * the particle.
   */
  inline void execute() {
    auto t0 = profile_timestamp();

    // reset the device count for cell ids that need mapping
    auto k_lookup_count = this->d_lookup_count.ptr;
    auto reset_event = this->sycl_target.queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<>([=]() { k_lookup_count[0] = 0; });
    });

    const auto npart_local = this->mpi_rank_dat->get_npart_local();
    this->d_lookup_local_cells.realloc_no_copy(npart_local);
    this->d_lookup_local_layers.realloc_no_copy(npart_local);

    reset_event.wait();

    // pointers to access dats in kernel
    auto k_position_dat = this->position_dat->cell_dat.device_ptr();
    auto k_mpi_rank_dat = this->mpi_rank_dat->cell_dat.device_ptr();

    // iteration set
    auto pl_iter_range = this->mpi_rank_dat->get_particle_loop_iter_range();
    auto pl_stride = this->mpi_rank_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->mpi_rank_dat->get_particle_loop_npart_cell();

    // pointers to access BufferDevices in the kernel
    auto k_lookup_local_cells = this->d_lookup_local_cells.ptr;
    auto k_lookup_local_layers = this->d_lookup_local_layers.ptr;

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                // Inspect to see if the a mpi rank has been identified for a
                // local communication pattern.
                const auto mpi_rank_on_dat = k_mpi_rank_dat[cellx][1][layerx];
                if (mpi_rank_on_dat < 0) {
                  // Atomically increment the lookup count
                  sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      atomic_count(k_lookup_count[0]);
                  const int index = atomic_count.fetch_add(1);

                  // store this particles location so that it can be
                  // directly accessed later
                  k_lookup_local_cells[index] = cellx;
                  k_lookup_local_layers[index] = layerx;
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->sycl_target.queue
        .memcpy(this->h_lookup_count.ptr, this->d_lookup_count.ptr,
                this->d_lookup_count.size_bytes())
        .wait();
    const auto npart_query = this->h_lookup_count.ptr[0];

    // these global indices are passed to the mesh heirarchy toget the rank
    this->d_lookup_global_cells.realloc_no_copy(npart_query * 6);
    this->h_lookup_global_cells.realloc_no_copy(npart_query * 6);
    this->h_lookup_ranks.realloc_no_copy(npart_query);
    this->d_lookup_ranks.realloc_no_copy(npart_query);

    auto k_lookup_global_cells = this->d_lookup_global_cells.ptr;
    auto k_lookup_ranks = this->d_lookup_ranks.ptr;

    // variables required in the kernel to map positions to cells
    auto mesh_heirarchy = this->h_mesh.get_mesh_hierarchy();
    auto k_ndim = mesh_heirarchy->ndim;
    const REAL k_inverse_cell_width_coarse =
        mesh_heirarchy->inverse_cell_width_coarse;
    const REAL k_inverse_cell_width_fine =
        mesh_heirarchy->inverse_cell_width_fine;
    const REAL k_cell_width_coarse = mesh_heirarchy->cell_width_coarse;
    const REAL k_cell_width_fine = mesh_heirarchy->cell_width_fine;
    const REAL k_ncells_dim_fine = mesh_heirarchy->ncells_dim_fine;

    for (int dimx = 0; dimx < k_ndim; dimx++) {
      this->h_origin.ptr[dimx] = mesh_heirarchy->origin[dimx];
      this->h_dims.ptr[dimx] = mesh_heirarchy->dims[dimx];
    }
    this->sycl_target.queue
        .memcpy(this->d_origin.ptr, this->h_origin.ptr,
                this->h_origin.size_bytes())
        .wait();
    this->sycl_target.queue
        .memcpy(this->d_dims.ptr, this->h_dims.ptr, this->h_dims.size_bytes())
        .wait();

    auto k_origin = this->d_origin.ptr;
    auto k_dims = this->d_dims.ptr;

    // map particles positions to coarse and fine cells in the mesh heirarchy
    this->sycl_target.queue
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
                if ((ABS(pos - k_dims[dimx] * k_cell_width_coarse) /
                     ABS(pos)) <= tol) {
                  cell_coarse = k_dims[dimx] - 1;
                  k_lookup_global_cells[(idx * k_ndim * 2) + dimx] =
                      cell_coarse;
                } else {
                  cell_coarse = 0;
                  k_lookup_global_cells[(idx * k_ndim * 2) + dimx] = -2;
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
                  k_lookup_global_cells[(idx * k_ndim * 2) + dimx + k_ndim] =
                      -2;
                }
              } else {
                k_lookup_global_cells[(idx * k_ndim * 2) + dimx + k_ndim] =
                    cell_fine;
              }
            }
          });
        })
        .wait_and_throw();

    // copy the computed indicies to the host
    if (this->d_lookup_global_cells.size_bytes() > 0) {
      this->sycl_target.queue
          .memcpy(this->h_lookup_global_cells.ptr,
                  this->d_lookup_global_cells.ptr,
                  this->d_lookup_global_cells.size_bytes())
          .wait_and_throw();
    }
    // get the mpi ranks
    mesh_heirarchy->get_owners(npart_query, this->h_lookup_global_cells.ptr,
                               this->h_lookup_ranks.ptr);

    // copy the mpi ranks back to the ParticleDat
    if (this->h_lookup_ranks.size_bytes() > 0) {
      this->sycl_target.queue
          .memcpy(this->d_lookup_ranks.ptr, this->h_lookup_ranks.ptr,
                  this->h_lookup_ranks.size_bytes())
          .wait_and_throw();
    }

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(sycl::range<1>(npart_query), [=](sycl::id<1> idx) {
            const INT cellx = k_lookup_local_cells[idx];
            const INT layerx = k_lookup_local_layers[idx];
            k_mpi_rank_dat[cellx][0][layerx] = k_lookup_ranks[idx];
          });
        })
        .wait_and_throw();

    sycl_target.profile_map.inc("MeshHierarchyGlobalMap", "execute", 1,
                                profile_elapsed(t0, profile_timestamp()));
  };
};

/**
 *  Set all components 0 and 1 of particles to -1 in the passed ParticleDat.
 *
 *  @param mpi_rank_dat ParticleDat containing MPI ranks to reset.
 */
inline void reset_mpi_ranks(ParticleDatShPtr<INT> &mpi_rank_dat) {

  // iteration set
  auto pl_iter_range = mpi_rank_dat->get_particle_loop_iter_range();
  auto pl_stride = mpi_rank_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell = mpi_rank_dat->get_particle_loop_npart_cell();

  // pointers to access BufferDevices in the kernel
  auto k_mpi_rank_dat = mpi_rank_dat->cell_dat.device_ptr();
  auto sycl_target = mpi_rank_dat->sycl_target;
  sycl_target.queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

          k_mpi_rank_dat[cellx][0][layerx] = -1;
          k_mpi_rank_dat[cellx][1][layerx] = -1;
          NESO_PARTICLES_KERNEL_END
        });
      })
      .wait_and_throw();
};

} // namespace NESO::Particles
#endif
