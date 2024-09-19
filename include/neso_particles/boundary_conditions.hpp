#ifndef _NESO_PARTICLES_BOUNDARY_CONDITIONS
#define _NESO_PARTICLES_BOUNDARY_CONDITIONS

#include <cmath>
#include <memory>

#include "domain.hpp"
#include "loop/particle_loop.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 * Periodic boundary conditions implementation designed to work with a
 * CartesianHMesh.
 */
class CartesianPeriodic {
private:
  BufferDevice<REAL> d_extents;
  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<CartesianHMesh> mesh;
  ParticleDatSharedPtr<REAL> position_dat;
  ParticleLoopSharedPtr pbc_loop;

public:
  /// Disable (implicit) copies.
  CartesianPeriodic(const CartesianPeriodic &st) = delete;
  /// Disable (implicit) copies.
  CartesianPeriodic &operator=(CartesianPeriodic const &a) = delete;

  ~CartesianPeriodic(){};

  /**
   * Construct instance to apply periodic boundary conditions to particles
   * within the passed ParticleDat.
   *
   * @param sycl_target SYCLTarget to use as compute device.
   * @param mesh CartedianHMesh instance to use a domain for the particles.
   * @param position_dat ParticleDat containing particle positions.
   */
  CartesianPeriodic(SYCLTargetSharedPtr sycl_target,
                    std::shared_ptr<CartesianHMesh> mesh,
                    ParticleDatSharedPtr<REAL> position_dat)
      : mesh(mesh), sycl_target(sycl_target), d_extents(sycl_target, 3),
        position_dat(position_dat) {

    NESOASSERT(mesh->ndim <= 3, "bad mesh ndim");
    BufferHost<REAL> h_extents(sycl_target, 3);
    for (int dimx = 0; dimx < mesh->ndim; dimx++) {
      h_extents.ptr[dimx] = this->mesh->global_extents[dimx];
    }
    sycl_target->queue
        .memcpy(this->d_extents.ptr, h_extents.ptr, mesh->ndim * sizeof(REAL))
        .wait_and_throw();

    const int k_ndim = this->mesh->ndim;
    NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
    const REAL *RESTRICT k_extents = this->d_extents.ptr;

    this->pbc_loop = particle_loop(
        "CartesianPeriodicPBC", position_dat,
        [=](auto P) {
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const REAL pos = P[dimx];
            // offset the position in the current dimension to be
            // positive by adding a value times the extent
            const int n_extent_offset_int = abs((int)pos);
            const REAL tmp_extent = k_extents[dimx];
            const REAL n_extent_offset_real = n_extent_offset_int + 2;
            const REAL pos_fmod =
                sycl::fmod(pos + n_extent_offset_real * tmp_extent, tmp_extent);
            P[dimx] = pos_fmod;
          }
        },
        Access::write(position_dat));
  };

  /**
   * Construct instance to apply periodic boundary conditions to particles
   * within the passed ParticleDat.
   *
   * @param mesh CartedianHMesh instance to use a domain for the particles.
   * @param particle_group ParticleGroup to apply periodic boundary conditions
   * to.
   */
  CartesianPeriodic(std::shared_ptr<CartesianHMesh> mesh,
                    ParticleGroupSharedPtr particle_group)
      : CartesianPeriodic(particle_group->sycl_target, mesh,
                          particle_group->position_dat){

        };

  /**
   * Apply periodic boundary conditions to the particle positions in the
   * ParticleDat this instance was created with.
   */
  inline void execute() { this->pbc_loop->execute(); }
};

} // namespace NESO::Particles

#endif
