#ifndef _NESO_PARTICLES_CELL_BINNING
#define _NESO_PARTICLES_CELL_BINNING

#include <CL/sycl.hpp>
#include <cmath>

#include "domain.hpp"
#include "loop/particle_loop.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;

namespace NESO::Particles {

/**
 * Bin particle positions into the cells of a CartesianHMesh.
 */
class CartesianCellBin {
private:
  BufferDevice<int> d_cell_counts;
  BufferDevice<int> d_cell_starts;
  BufferDevice<int> d_cell_ends;

  SYCLTargetSharedPtr sycl_target;
  CartesianHMeshSharedPtr mesh;
  ParticleDatSharedPtr<REAL> position_dat;
  ParticleDatSharedPtr<INT> cell_id_dat;
  ParticleLoopSharedPtr loop;

public:
  /// Disable (implicit) copies.
  CartesianCellBin(const CartesianCellBin &st) = delete;
  /// Disable (implicit) copies.
  CartesianCellBin &operator=(CartesianCellBin const &a) = delete;

  ~CartesianCellBin(){};

  /**
   * Create instance to bin particles into cells of a CartesianHMesh.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param mesh CartesianHMeshSharedPtr to containing the particles.
   * @param position_dat ParticleDat with components equal to the mesh dimension
   * containing particle positions.
   * @param cell_id_dat ParticleDat to write particle cell ids to.
   */
  CartesianCellBin(SYCLTargetSharedPtr sycl_target,
                   CartesianHMeshSharedPtr mesh,
                   ParticleDatSharedPtr<REAL> position_dat,
                   ParticleDatSharedPtr<INT> cell_id_dat)
      : sycl_target(sycl_target), mesh(mesh), position_dat(position_dat),
        cell_id_dat(cell_id_dat), d_cell_counts(sycl_target, 3),
        d_cell_starts(sycl_target, 3), d_cell_ends(sycl_target, 3) {

    NESOASSERT(mesh->ndim <= 3, "bad mesh ndim");
    BufferHost<int> h_cell_counts(sycl_target, 3);
    for (int dimx = 0; dimx < mesh->ndim; dimx++) {
      h_cell_counts.ptr[dimx] = this->mesh->cell_counts_local[dimx];
    }
    sycl_target->queue
        .memcpy(this->d_cell_counts.ptr, h_cell_counts.ptr,
                mesh->ndim * sizeof(int))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(this->d_cell_starts.ptr, mesh->cell_starts,
                mesh->ndim * sizeof(int))
        .wait_and_throw();
    sycl_target->queue
        .memcpy(this->d_cell_ends.ptr, mesh->cell_ends,
                mesh->ndim * sizeof(int))
        .wait_and_throw();

    const int k_ndim = this->mesh->ndim;

    NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
    auto k_inverse_cell_width_fine = this->mesh->inverse_cell_width_fine;
    auto k_cell_width_fine = this->mesh->cell_width_fine;

    auto k_cell_counts = this->d_cell_counts.ptr;
    auto k_cell_starts = this->d_cell_starts.ptr;
    auto k_cell_ends = this->d_cell_ends.ptr;

    this->loop = particle_loop(
        "CartesianCellBin", position_dat->get_particle_group(),
        [=](auto positions, auto cell_id) {
          int cell_tmps[3];

          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const REAL pos =
                positions[dimx] - k_cell_starts[dimx] * k_cell_width_fine;
            int cell_tmp = ((REAL)pos * k_inverse_cell_width_fine);
            cell_tmp = (cell_tmp < 0) ? 0 : cell_tmp;
            cell_tmp = (cell_tmp >= k_cell_ends[dimx]) ? k_cell_ends[dimx] - 1
                                                       : cell_tmp;
            cell_tmps[dimx] = cell_tmp;
          }

          // convert to linear index
          int linear_index = cell_tmps[k_ndim - 1];
          for (int dimx = k_ndim - 2; dimx >= 0; dimx--) {
            linear_index *= k_cell_counts[dimx];
            linear_index += cell_tmps[dimx];
          }
          cell_id[0] = linear_index;
        },
        Access::read(position_dat->sym), Access::write(cell_id_dat->sym));
  };

  /**
   *  Apply the cell binning kernel to each particle stored on this MPI rank.
   *  Particles must be within the domain region owned by this MPI rank.
   */
  inline void execute() { this->loop->execute(); }
};

} // namespace NESO::Particles

#endif
