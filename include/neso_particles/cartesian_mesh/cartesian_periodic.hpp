#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_PERIODIC_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_PERIODIC_HPP_

#include <cmath>
#include <memory>

#include "../domain.hpp"
#include "../loop/particle_loop.hpp"
#include "../particle_dat.hpp"
#include "../profiling.hpp"
#include "../sycl_typedefs.hpp"
#include "../typedefs.hpp"
#include "cartesian_h_mesh.hpp"

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

  ~CartesianPeriodic() = default;

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
                    ParticleDatSharedPtr<REAL> position_dat);

  /**
   * Construct instance to apply periodic boundary conditions to particles
   * within the passed ParticleDat.
   *
   * @param mesh CartedianHMesh instance to use a domain for the particles.
   * @param particle_group ParticleGroup to apply periodic boundary conditions
   * to.
   */
  CartesianPeriodic(std::shared_ptr<CartesianHMesh> mesh,
                    ParticleGroupSharedPtr particle_group);

  /**
   * Apply periodic boundary conditions to the particle positions in the
   * ParticleDat this instance was created with.
   */
  void execute();
};

} // namespace NESO::Particles

#endif
