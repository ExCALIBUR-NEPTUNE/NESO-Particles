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
  std::unique_ptr<BufferDeviceHost<INT>> dh_cells;
  std::unique_ptr<BufferDeviceHost<INT>> dh_ranks;

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
    this->dh_cells->realloc_no_copy(npart_local);
    this->dh_ranks->realloc_no_copy(npart_local);

    auto dat_positions = particle_group.position_dat;
    auto k_interlaced_positions = this->d_interlaced_positions->ptr;
    auto interlace_loop = particle_loop(
        "DMPlexLocalMapper::interlace", dat_positions,
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
    PetscSF cell_sf = nullptr;
    // PETSCCHK(PetscSFCreate(MPI_COMM_SELF, &cell_sf));
    PETSCCHK(DMLocatePoints(dm, v, DM_POINTLOCATION_NONE, &cell_sf));
    const PetscSFNode *cells;
    PetscInt n_found;
    const PetscInt *found;
    PETSCCHK(PetscSFGetGraph(cell_sf, NULL, &n_found, &found, &cells));

    for (int px = 0; px < npart_local; px++) {
      const auto px_cell = cells[px].index;
      // If the cell is negative then PETSc did not find a cell containing the
      // point.
      this->dh_cells->h_buffer.ptr[px] = (px_cell > -1) ? px_cell : -1;
    }

    // TODO check halo cells

    // Write cells back to particles
    this->dh_cells->host_to_device();
    this->dh_ranks->host_to_device();

    auto k_cells = this->dh_cells->d_buffer.ptr;
    auto k_ranks = this->dh_ranks->d_buffer.ptr;
    auto dat_cells = particle_group.cell_id_dat;
    auto dat_ranks = particle_group.mpi_rank_dat;

    auto deinterlace_loop = particle_loop(
        "DMPlexLocalMapper::deinterlace", dat_positions,
        [=](auto INDEX, auto CELL, auto RANK) {
          const auto linear_index = INDEX.get_loop_linear_index();
          CELL.at(0) = k_cells[linear_index];
          RANK.at(1) = k_ranks[linear_index];
        },
        Access::read(ParticleLoopIndex{}), Access::write(dat_cells),
        Access::write(dat_ranks));

    if (map_cell > -1) {
      deinterlace_loop->execute(map_cell);
    } else {
      deinterlace_loop->execute();
    }

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
    this->dh_cells = std::make_unique<BufferDeviceHost<INT>>(sycl_target, 1);
    this->dh_ranks = std::make_unique<BufferDeviceHost<INT>>(sycl_target, 1);
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
