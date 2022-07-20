#ifndef _NESO_PARTICLES_CELL_BINNING
#define _NESO_PARTICLES_CELL_BINNING

#include <CL/sycl.hpp>
#include <cmath>

#include "domain.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
using namespace std;

namespace NESO::Particles {

class CartesianCellBin {
private:
  BufferDevice<int> d_cell_counts;
  BufferDevice<int> d_cell_starts;
  BufferDevice<int> d_cell_ends;

public:
  SYCLTarget &sycl_target;
  CartesianHMesh &mesh;
  ParticleDatShPtr<REAL> position_dat;
  ParticleDatShPtr<INT> cell_id_dat;

  ~CartesianCellBin(){};
  CartesianCellBin(SYCLTarget &sycl_target, CartesianHMesh &mesh,
                   ParticleDatShPtr<REAL> position_dat,
                   ParticleDatShPtr<INT> cell_id_dat)
      : sycl_target(sycl_target), mesh(mesh), position_dat(position_dat),
        cell_id_dat(cell_id_dat), d_cell_counts(sycl_target, 3),
        d_cell_starts(sycl_target, 3), d_cell_ends(sycl_target, 3) {

    NESOASSERT(mesh.ndim <= 3, "bad mesh ndim");
    BufferHost<int> h_cell_counts(sycl_target, 3);
    for (int dimx = 0; dimx < mesh.ndim; dimx++) {
      h_cell_counts.ptr[dimx] = this->mesh.cell_counts_local[dimx];
    }
    sycl_target.queue
        .memcpy(this->d_cell_counts.ptr, h_cell_counts.ptr,
                mesh.ndim * sizeof(int))
        .wait_and_throw();
    sycl_target.queue
        .memcpy(this->d_cell_starts.ptr, mesh.cell_starts,
                mesh.ndim * sizeof(int))
        .wait_and_throw();
    sycl_target.queue
        .memcpy(this->d_cell_ends.ptr, mesh.cell_ends, mesh.ndim * sizeof(int))
        .wait_and_throw();
  };

  inline void execute() {
    auto t0 = profile_timestamp();

    auto pl_iter_range = this->position_dat->get_particle_loop_iter_range();
    auto pl_stride = this->position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->position_dat->get_particle_loop_npart_cell();
    const int k_ndim = this->mesh.ndim;

    NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
    auto k_positions_dat = this->position_dat->cell_dat.device_ptr();
    auto k_cell_id_dat = this->cell_id_dat->cell_dat.device_ptr();
    auto k_inverse_cell_width_fine = this->mesh.inverse_cell_width_fine;
    auto k_cell_width_fine = this->mesh.cell_width_fine;

    auto k_cell_counts = this->d_cell_counts.ptr;
    auto k_cell_starts = this->d_cell_starts.ptr;
    auto k_cell_ends = this->d_cell_ends.ptr;

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                int cell_tmps[3];

                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  const REAL pos = k_positions_dat[cellx][dimx][layerx] -
                                   k_cell_starts[dimx] * k_cell_width_fine;
                  int cell_tmp = ((REAL)pos * k_inverse_cell_width_fine);
                  cell_tmp = (cell_tmp < 0) ? 0 : cell_tmp;
                  cell_tmp = (cell_tmp >= k_cell_ends[dimx])
                                 ? k_cell_ends[dimx] - 1
                                 : cell_tmp;
                  cell_tmps[dimx] = cell_tmp;
                }

                // convert to linear index
                int linear_index = cell_tmps[k_ndim - 1];
                for (int dimx = k_ndim - 2; dimx >= 0; dimx--) {
                  linear_index *= k_cell_counts[dimx];
                  linear_index += cell_tmps[dimx];
                }
                k_cell_id_dat[cellx][0][layerx] = linear_index;
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target.profile_map.inc("CartesianCellBin", "execute", 1,
                                profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO::Particles

#endif
