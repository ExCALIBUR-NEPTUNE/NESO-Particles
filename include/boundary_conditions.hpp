#ifndef _NESO_PARTICLES_BOUNDARY_CONDITIONS
#define _NESO_PARTICLES_BOUNDARY_CONDITIONS

#include <CL/sycl.hpp>
#include <cmath>

#include "domain.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
using namespace std;

namespace NESO::Particles {

class CartesianPeriodic {
private:
  BufferDevice<REAL> d_extents;

public:
  SYCLTarget &sycl_target;
  CartesianHMesh &mesh;
  ParticleDatShPtr<REAL> position_dat;

  ~CartesianPeriodic(){};
  CartesianPeriodic(SYCLTarget &sycl_target, CartesianHMesh &mesh,
                    ParticleDatShPtr<REAL> position_dat)
      : sycl_target(sycl_target), mesh(mesh), position_dat(position_dat),
        d_extents(sycl_target, 3) {

    NESOASSERT(mesh.ndim <= 3, "bad mesh ndim");
    BufferHost<REAL> h_extents(sycl_target, 3);
    for (int dimx = 0; dimx < mesh.ndim; dimx++) {
      h_extents.ptr[dimx] = this->mesh.global_extents[dimx];
    }
    sycl_target.queue
        .memcpy(this->d_extents.ptr, h_extents.ptr, mesh.ndim * sizeof(REAL))
        .wait_and_throw();
  };

  inline void execute() {

    auto t0 = profile_timestamp();
    auto pl_iter_range = this->position_dat->get_particle_loop_iter_range();
    auto pl_stride = this->position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = this->position_dat->get_particle_loop_npart_cell();
    const int k_ndim = this->mesh.ndim;

    NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
    auto k_extents = this->d_extents.ptr;
    auto k_positions_dat = this->position_dat->cell_dat.device_ptr();

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  const REAL pos = k_positions_dat[cellx][dimx][layerx];
                  // offset the position in the current dimension to be
                  // positive by adding an integer times the extent
                  const REAL n_extent_offset_real = ABS(pos / k_extents[dimx]);
                  const INT n_extent_offset_int = n_extent_offset_real + 2.0;
                  const REAL pos_fmod =
                      fmod(pos + n_extent_offset_int * k_extents[dimx],
                           k_extents[dimx]);
                  k_positions_dat[cellx][dimx][layerx] = pos_fmod;
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target.profile_map.inc("CartesianPeriodic", "execute", 1,
                                profile_elapsed(t0, profile_timestamp()));
  }
};

} // namespace NESO::Particles

#endif
