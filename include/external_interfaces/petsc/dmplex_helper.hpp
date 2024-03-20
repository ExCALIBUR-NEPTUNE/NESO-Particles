#ifndef _NESO_PARTICLES_DMPLEX_HELPER_HPP_
#define _NESO_PARTICLES_DMPLEX_HELPER_HPP_

#include "../common/bounding_box.hpp"
#include "petsc_common.hpp"
#include <limits>
#include <memory>
#include <vector>

namespace NESO::Particles::PetscInterface {

/**
 * If there are more than 1 MPI ranks distribute the mesh.
 *
 * @param[in, out] dm DMPlex to distribute, original DMPlex is destroyed.
 * @param[in] comm MPI communicator, default MPI_COMM_WORLD.
 */
inline void generic_distribute(DM *dm, MPI_Comm comm = MPI_COMM_WORLD) {
  int size;
  MPICHK(MPI_Comm_size(comm, &size));
  if (size > 1) {
    DM dm_out;
    PETSCCHK(DMPlexDistribute(*dm, 0, nullptr, &dm_out));
    NESOASSERT(dm_out, "Could not distribute mesh.");
    PETSCCHK(DMDestroy(dm));
    *dm = dm_out;
  }
}

class DMPlexHelper {
protected:
  PetscInt cell_start;
  PetscInt cell_end;

  inline void check_valid_cell(const PetscInt cell) const {
    NESOASSERT((cell > -1) && (cell < (this->cell_end - this->cell_start)),
               "Bad cell index passed.");
  }

public:
  MPI_Comm comm;
  DM dm;
  PetscInt ndim;
  IS global_cell_numbers;

  /**
   * TODO
   */
  DMPlexHelper(MPI_Comm comm, DM dm) : comm(comm), dm(dm) {
    DMPlexInterpolatedFlag interpolated;
    PETSCCHK(DMPlexIsInterpolated(this->dm, &interpolated));
    NESOASSERT(interpolated == DMPLEX_INTERPOLATED_FULL,
               "Expected fully interpolated mesh.");
    PETSCCHK(DMGetCoordinateDim(this->dm, &this->ndim));
    PETSCCHK(DMPlexGetHeightStratum(this->dm, 0, &this->cell_start,
                                    &this->cell_end));
    PETSCCHK(DMPlexGetCellNumbering(this->dm, &this->global_cell_numbers));
  }

  /**
   * Get the global index of a cell from the local cell index.
   *
   * @param cell Local cell index.
   * @returns Global cell index.
   */
  inline PetscInt get_cell_global_index(const PetscInt cell) {
    this->check_valid_cell(cell);

    PetscInt global_cell;
    const PetscInt *ptr;
    PETSCCHK(ISGetIndices(this->global_cell_numbers, &ptr));
    global_cell = ptr[cell];
    PETSCCHK(ISRestoreIndices(this->global_cell_numbers, &ptr));
    NESOASSERT(global_cell > -1,
               "Cell index was negative indicating a remote cell.");

    return global_cell;
  }

  /**
   * Get a bounding box for a mesh cell (assumes linear mesh).
   *
   * @param cell Local cell index.
   * @returns Bounding box for cell.
   */
  inline ExternalCommon::BoundingBoxSharedPtr
  get_cell_bounding_box(const PetscInt cell) {
    this->check_valid_cell(cell);

    std::vector<REAL> bb = {
        std::numeric_limits<REAL>::max(), std::numeric_limits<REAL>::max(),
        std::numeric_limits<REAL>::max(), std::numeric_limits<REAL>::min(),
        std::numeric_limits<REAL>::min(), std::numeric_limits<REAL>::min()};

    for (int dx = this->ndim; dx < 3; dx++) {
      bb[dx] = 0.0;
      bb[dx + 3] = 0.0;
    }

    const PetscScalar *array;
    PetscScalar *coords = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;
    PETSCCHK(DMPlexGetCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                      &num_coords, &array, &coords));
    NESOASSERT(coords != nullptr, "No vertices returned for cell.");
    const PetscInt num_verts = num_coords / ndim;
    for (PetscInt vx = 0; vx < num_verts; vx++) {
      for (PetscInt dimx = 0; dimx < this->ndim; dimx++) {
        const REAL cx = coords[vx * ndim + dimx];
        bb[dimx] = std::min(bb[dimx], cx);
        bb[dimx + 3] = std::max(bb[dimx + 3], cx);
      }
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                          &num_coords, &array, &coords));
    return std::make_shared<ExternalCommon::BoundingBox>(bb);
  }

  /**
   * Get average of the vertices of a cell.
   *
   * @param[in] cell Local cell index.
   * @param[in, out] average Vector of average of vertices.
   */
  inline void get_cell_vertex_average(const PetscInt cell,
                                      std::vector<REAL> &average) {
    this->check_valid_cell(cell);
    NESOASSERT(average.size() == this->ndim,
               "Missmatch between vector size and number of dimensions");

    std::fill(average.begin(), average.end(), 0.0);
    const PetscScalar *array;
    PetscScalar *coords = nullptr;
    PetscInt num_coords;
    PetscBool is_dg;
    PETSCCHK(DMPlexGetCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                      &num_coords, &array, &coords));
    NESOASSERT(coords != nullptr, "No vertices returned for cell.");
    const PetscInt num_verts = num_coords / ndim;
    for (PetscInt vx = 0; vx < num_verts; vx++) {
      for (PetscInt dimx = 0; dimx < this->ndim; dimx++) {
        const REAL cx = coords[vx * ndim + dimx];
        average.at(dimx) += cx;
      }
    }
    PETSCCHK(DMPlexRestoreCellCoordinates(dm, this->cell_start + cell, &is_dg,
                                          &num_coords, &array, &coords));

    const REAL tmp_factor = 1.0 / ((REAL)num_verts);
    for (int dx = 0; dx < this->ndim; dx++) {
      average.at(dx) *= tmp_factor;
    }
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
