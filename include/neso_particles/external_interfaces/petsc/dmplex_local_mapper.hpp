#ifndef _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_
#define _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_

#include "../../compute_target.hpp"
#include "../../local_mapping.hpp"
#include "../../loop/particle_loop.hpp"
#include "../../particle_group.hpp"
#include "dmplex_interface.hpp"
#include "particle_cell_mapping/dmplex_2d_mapper.hpp"
#include "particle_cell_mapping/dmplex_host_mapper.hpp"
#include <deque>

namespace NESO::Particles::PetscInterface {

/**
 * Top level interface for mapping particles into DMPlex cells.
 */
class DMPlexLocalMapper : public LocalMapper {
protected:
  std::unique_ptr<DMPlexHostMapper> mapper_host;
  std::unique_ptr<DMPlex2DMapper> mapper_2d;
  int ndim;

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;
  virtual ~DMPlexLocalMapper() = default;

  /**
   * Create mappers for a DMPlex.
   *
   * @param sycl_target Compute target for mappers.
   * @param dmplex_interface DMPlexInterface for which to create particle to
   * cell mappers for.
   */
  DMPlexLocalMapper(SYCLTargetSharedPtr sycl_target,
                    DMPlexInterfaceSharedPtr dmplex_interface)
      : sycl_target(sycl_target), dmplex_interface(dmplex_interface) {

    this->ndim = dmplex_interface->get_ndim();
    if (this->ndim == 2) {
      this->mapper_2d =
          std::make_unique<DMPlex2DMapper>(sycl_target, dmplex_interface);
    } else {
      this->mapper_host =
          std::make_unique<DMPlexHostMapper>(sycl_target, dmplex_interface);
    }
  }

  /**
   * This function maps particle positions to cells on the underlying mesh.
   *
   * @param particle_group ParticleGroup containing particle positions.
   */
  virtual inline void map(ParticleGroup &particle_group,
                          const int map_cell = -1) override {
    if (this->mapper_2d) {
      this->mapper_2d->map(particle_group, map_cell);
    } else {
      this->mapper_host->map(particle_group, map_cell);
    }
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
