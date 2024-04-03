#ifndef _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_
#define _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_

#include "../../compute_target.hpp"
#include "../../local_mapping.hpp"
#include "../../loop/particle_loop.hpp"
#include "../../particle_group.hpp"
#include "dmplex_interface.hpp"

namespace NESO::Particles::PetscInterface {

class DMPlexLocalMapper : public LocalMapper {
protected:
  std::unique_ptr<BufferDevice<PetscScalar>> d_interlaced_positions;

  inline void map_host(ParticleGroup &particle_group, const int map_cell) {

    PetscInt tmp_petsc_int;
    DM &dm = this->dmplex_interface->dmh->dm;
    PETSCCHK(DMGetCoordinateDim(dm, &tmp_petsc_int));
    const PetscInt ndim_coord = tmp_petsc_int;

    // copy all particles into Vec
    const auto npart_local = (map_cell > -1)
                                 ? particle_group.get_npart_cell(map_cell)
                                 : particle_group.get_npart_local();

    const PetscInt local_ncomp =
        static_cast<PetscInt>(npart_local) * ndim_coord;
    this->d_interlaced_positions->realloc_no_copy(local_ncomp);

    auto dat_positions = particle_group.position_dat;
    auto k_interlaced_positions = this->d_interlaced_positions->ptr;
    auto interlace_loop = particle_loop(
        "DMPlexLocalMapper", dat_positions,
        [=](auto INDEX, auto P) {
          const auto linear_index = INDEX.get_loop_linear_index();
          for (int dx = 0; dx < ndim_coord; dx++) {
            k_interlaced_positions[linear_index * ndim_coord + dx] = P.at(dx);
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(dat_positions));

    if (map_cell > -1) {
      interlace_loop->submit(map_cell);
    } else {
      interlace_loop->submit();
    }

    // Create a Vec for the positions
    Vec v;
    PETSCCHK(VecCreate(MPI_COMM_SELF, &v));
    PETSCCHK(VecSetSizes(v, local_ncomp, local_ncomp));
    PETSCCHK(VecSetBlockSize(v, ndim_coord));
    PETSCCHK(VecSetFromOptions(v));

    PetscScalar *v_ptr;
    PETSCCHK(VecGetArrayWrite(v, &v_ptr));

    // copy the positions from device to host
    interlace_loop->wait();
    sycl_target->queue
        .memcpy(v_ptr, k_interlaced_positions,
                local_ncomp * sizeof(PetscScalar))
        .wait_and_throw();

    PETSCCHK(VecRestoreArrayWrite(v, &v_ptr));
    PETSCCHK(VecView(v, PETSC_VIEWER_STDOUT_SELF));

    // call DMLocatePoints with Vec
    
    //TODO

    // Write cells back to particles
    
    //TODO

    PETSCCHK(VecDestroy(&v));
  }

public:
  SYCLTargetSharedPtr sycl_target;
  DMPlexInterfaceSharedPtr dmplex_interface;

  /**
   * TODO
   */
  DMPlexLocalMapper(SYCLTargetSharedPtr sycl_target,
                    DMPlexInterfaceSharedPtr dmplex_interface)
      : sycl_target(sycl_target), dmplex_interface(dmplex_interface) {
    this->d_interlaced_positions =
        std::make_unique<BufferDevice<PetscScalar>>(sycl_target, 1);
  }

  /**
   * This function maps particle positions to cells on the underlying mesh.
   *
   * @param particle_group ParticleGroup containing particle positions.
   */
  virtual inline void map(ParticleGroup &particle_group,
                          const int map_cell = -1) override {
    this->map_host(particle_group, map_cell);
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
