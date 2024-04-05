#ifndef _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_
#define _NESO_PARTICLES_DMPLEX_LOCAL_MAPPER_HPP_

#include "../../compute_target.hpp"
#include "../../local_mapping.hpp"
#include "../../loop/particle_loop.hpp"
#include "../../particle_group.hpp"
#include "dmplex_interface.hpp"
#include <deque>

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
    const auto rank = this->sycl_target->comm_pair.rank_parent;

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

    // call DMLocatePoints with Vec
    PetscSF cell_sf = nullptr;
    PETSCCHK(DMLocatePoints(dm, v, DM_POINTLOCATION_NONE, &cell_sf));
    const PetscSFNode *cells;
    PetscInt n_found;
    const PetscInt *found;
    PETSCCHK(PetscSFGetGraph(cell_sf, NULL, &n_found, &found, &cells));

    std::deque<int> not_local_found;
    for (int px = 0; px < npart_local; px++) {
      const auto px_cell = cells[px].index;
      // If the cell is negative then PETSc did not find a cell containing the
      // point.
      const bool cell_found = px_cell > -1;
      this->dh_cells->h_buffer.ptr[px] = (cell_found) ? px_cell : -1;
      this->dh_ranks->h_buffer.ptr[px] = (cell_found) ? rank : -1;
      if (!cell_found) {
        not_local_found.push_back(px);
      }
    }

    if (this->dmplex_interface->dmh_halo) {
      DM &dm_halo = this->dmplex_interface->dmh_halo->dm;
      // Look in halo cells for particles not found in local cells
      const int npart_not_found = not_local_found.size();
      // Create a Vec for the positions
      Vec v_halo;
      PETSCCHK(VecCreate(MPI_COMM_SELF, &v_halo));
      PETSCCHK(VecSetSizes(v_halo, local_ncomp, local_ncomp));
      PETSCCHK(VecSetBlockSize(v_halo, ndim_coord));
      PETSCCHK(VecSetFromOptions(v_halo));

      // Copy the positions from the local vector into the halo vector
      PetscScalar *v_halo_ptr;
      const PetscScalar *v_const_ptr;
      PETSCCHK(VecGetArrayRead(v, &v_const_ptr));
      PETSCCHK(VecGetArrayWrite(v_halo, &v_halo_ptr));
      for (int qx = 0; qx < npart_not_found; qx++) {
        const int px = not_local_found.at(qx);
        for (int dimx = 0; dimx < ndim_coord; dimx++) {
          v_halo_ptr[qx * ndim_coord + dimx] =
              v_const_ptr[px * ndim_coord + dimx];
        }
      }
      PETSCCHK(VecRestoreArrayWrite(v_halo, &v_halo_ptr));
      PETSCCHK(VecRestoreArrayRead(v, &v_const_ptr));

      // Call DMLocatePoints with the halo vector
      cell_sf = nullptr;
      PETSCCHK(
          DMLocatePoints(dm_halo, v_halo, DM_POINTLOCATION_NONE, &cell_sf));
      PETSCCHK(PetscSFGetGraph(cell_sf, NULL, &n_found, &found, &cells));
      for (int qx = 0; qx < npart_not_found; qx++) {
        const int px = not_local_found.at(qx);
        const auto px_cell = cells[qx].index;
        const bool cell_found = px_cell > -1;

        // Get the owning rank and local index of the cell on the remote rank.
        INT remote_cell, remote_rank;
        if (cell_found) {
          auto rank_cell =
              dmplex_interface->map_local_lid_remote_lid.at(px_cell);
          remote_rank = std::get<0>(rank_cell);
          remote_cell = std::get<1>(rank_cell);
        }
        this->dh_cells->h_buffer.ptr[px] = (cell_found) ? remote_cell : -1;
        this->dh_ranks->h_buffer.ptr[px] = (cell_found) ? remote_rank : -1;
      }

      PETSCCHK(VecDestroy(&v_halo));
    } else {
      NESOWARN(false, "Particles not mapped into cells.");
    }

    PETSCCHK(VecDestroy(&v));
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
