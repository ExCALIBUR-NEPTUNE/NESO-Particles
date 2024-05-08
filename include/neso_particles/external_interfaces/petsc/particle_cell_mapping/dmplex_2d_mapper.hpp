#ifndef _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_
#define _NESO_PARTICLES_DMPLEX_2D_MAPPER_H_

#include "../../../compute_target.hpp"
#include "../../../local_mapping.hpp"
#include "../../../loop/particle_loop.hpp"
#include "../../../particle_group.hpp"
#include "../dmplex_interface.hpp"

namespace NESO::Particles::PetscInterface {

class DMPlex2DMapper {
protected:

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;

  /**
   * TODO
   */
  DMPlex2DMapper(SYCLTargetSharedPtr sycl_target,
                   DMPlexInterfaceSharedPtr dmplex_interface)
      : sycl_target(sycl_target), dmplex_interface(dmplex_interface) {
  }

  /**
   * TODO
   */
  inline void map(ParticleGroup &particle_group, const int map_cell) {


  }
};

} // namespace NESO::Particles::PetscInterface

#endif
