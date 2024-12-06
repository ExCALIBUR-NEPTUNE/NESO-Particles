#ifndef _NESO_PARTICLES_PETSC_API_REDIRECTION_HPP_
#define _NESO_PARTICLES_PETSC_API_REDIRECTION_HPP_

#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscsys.h>

#ifndef NESO_PARTICLES_PETSC_VERSION_LT
#define NESO_PARTICLES_PETSC_VERSION_LT(x, y)                                  \
  ((PETSC_VERSION_MAJOR < (x)) ||                                              \
   ((PETSC_VERSION_MAJOR == (x)) && (PETSC_VERSION_MINOR < (y))))
#endif

namespace NESO::Particles::NPPETScAPI {

/**
 * Wraps DMPlexCreateBoxMesh to work pre and post petsc v3.22.
 */
inline PetscErrorCode NP_DMPlexCreateBoxMesh(
    MPI_Comm comm, PetscInt dim, PetscBool simplex, const PetscInt faces[],
    const PetscReal lower[], const PetscReal upper[],
    const DMBoundaryType periodicity[], PetscBool interpolate, DM *dm) {

#if NESO_PARTICLES_PETSC_VERSION_LT(3, 22)
  return DMPlexCreateBoxMesh(comm, dim, simplex, faces, lower, upper,
                             periodicity, interpolate, dm);
#else
  return DMPlexCreateBoxMesh(comm, dim, simplex, faces, lower, upper,
                             periodicity, interpolate, 0, PETSC_TRUE, dm);
#endif
}

}; // namespace NESO::Particles::NPPETScAPI

#endif
