#ifndef _NESO_PARTICLES_BOUNDARY_CONDITIONS
#define _NESO_PARTICLES_BOUNDARY_CONDITIONS

#include <CL/sycl.hpp>
#include <cmath>

#include "domain.hpp"
#include "loop/particle_loop.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"
#include <memory>

using namespace cl;
using namespace std;

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
      : CartesianPeriodic(mesh, position_dat->get_particle_group()){};

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
      : mesh(mesh), sycl_target(particle_group->sycl_target),
        d_extents(particle_group->sycl_target, 3) {

    NESOASSERT(mesh->ndim <= 3, "bad mesh ndim");
    BufferHost<REAL> h_extents(sycl_target, 3);
    for (int dimx = 0; dimx < mesh->ndim; dimx++) {
      h_extents.ptr[dimx] = this->mesh->global_extents[dimx];
    }
    sycl_target->queue
        .memcpy(this->d_extents.ptr, h_extents.ptr, mesh->ndim * sizeof(REAL))
        .wait_and_throw();

    auto position_sym = particle_group->position_dat->sym;
    const int k_ndim = this->mesh->ndim;
    NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
    const auto k_extents = this->d_extents.ptr;

    this->pbc_loop = particle_loop(
        "CartesianPeriodicPBC", particle_group,
        [=](auto P) {
          for (int dimx = 0; dimx < k_ndim; dimx++) {
            const REAL pos = P[dimx];
            // offset the position in the current dimension to be
            // positive by adding a value times the extent
            const REAL n_extent_offset_real = ABS(pos);
            const REAL tmp_extent = k_extents[dimx];
            const INT n_extent_offset_int = n_extent_offset_real + 2.0;
            const REAL pos_fmod =
                fmod(pos + n_extent_offset_int * tmp_extent, tmp_extent);
            P[dimx] = pos_fmod;
          }
        },
        Access::write(position_sym));
  };

  /**
   * Apply periodic boundary conditions to the particle positions in the
   * ParticleDat this instance was created with.
   */
  inline void execute() { this->pbc_loop->execute(); }
};

} // namespace NESO::Particles

#endif
