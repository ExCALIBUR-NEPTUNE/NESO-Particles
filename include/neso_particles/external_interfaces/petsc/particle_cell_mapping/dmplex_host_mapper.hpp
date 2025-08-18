#ifndef _NESO_PARTICLES_DMPLEX_HOST_MAPPER_H_
#define _NESO_PARTICLES_DMPLEX_HOST_MAPPER_H_

#include "../../../compute_target.hpp"
#include "../../../local_mapping.hpp"
#include "../../../loop/particle_loop_functions.hpp"
#include "../../../particle_group.hpp"
#include "../dmplex_interface.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Particle to cell mapping implementation that uses the host.
 */
class DMPlexHostMapper {
protected:
  std::unique_ptr<BufferDevice<PetscScalar>> d_interlaced_positions;
  std::unique_ptr<BufferDeviceHost<INT>> dh_cells;
  std::unique_ptr<BufferDeviceHost<INT>> dh_ranks;

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;

  /**
   * Create mapper for compute device and DMPlex using the host.
   *
   * @param sycl_target Compute device to create mapper on (the particles must
   * be on this compute device but the mapping occurs on the host).
   * @param dmplex_interface DMPlexInterface to create mapper for.
   */
  DMPlexHostMapper(SYCLTargetSharedPtr sycl_target,
                   DMPlexInterfaceSharedPtr dmplex_interface);

  /**
   * Map particles to cells on the host.
   *
   * @param particle_group Particles to map into cells.
   * @param map_cell Cell in particle group to determine cells for. Values less
   * than zero imply all particles should be mapped into cells.
   */
  void map(ParticleGroup &particle_group, const int map_cell);
};

} // namespace NESO::Particles::PetscInterface

#endif
