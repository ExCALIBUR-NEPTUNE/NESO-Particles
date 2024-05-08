#ifndef _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_
#define _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_

#include "../../compute_target.hpp"
#include "../../local_mapping.hpp"
#include "../../loop/particle_loop.hpp"
#include "../../particle_group.hpp"
#include "dmplex_interface.hpp"
#include "particle_cell_mapping/dmplex_host_mapper.hpp"
#include <deque>

namespace NESO::Particles::PetscInterface {

class DMPlexLocalMapper : public LocalMapper {
protected:
  std::unique_ptr<DMPlexHostMapper> host_mapper;

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;

  /**
   * TODO
   */
  DMPlexLocalMapper(SYCLTargetSharedPtr sycl_target,
                    DMPlexInterfaceSharedPtr dmplex_interface)
      : sycl_target(sycl_target), dmplex_interface(dmplex_interface) {
    this->host_mapper =
        std::make_unique<DMPlexHostMapper>(sycl_target, dmplex_interface);
  }

  /**
   * This function maps particle positions to cells on the underlying mesh.
   *
   * @param particle_group ParticleGroup containing particle positions.
   */
  virtual inline void map(ParticleGroup &particle_group,
                          const int map_cell = -1) override {
    this->host_mapper->map(particle_group, map_cell);
  }

  /**
   * Callback for ParticleGroup to execute for additional setup of the
   * LocalMapper that may involve the ParticleGroup.
   *
   * @param particle_group ParticleGroup instance.
   */
  virtual inline void particle_group_callback(
      [[maybe_unused]] ParticleGroup &particle_group) override {}
};

} // namespace NESO::Particles::PetscInterface

#endif
