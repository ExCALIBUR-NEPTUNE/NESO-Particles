#ifndef _NESO_PARTICLES_PETSC_API_REDIRECTION_HPP_
#define _NESO_PARTICLES_PETSC_API_REDIRECTION_HPP_

#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscsys.h>
#include <string>
#include <utility>

#ifndef NESO_PARTICLES_PETSC_VERSION_LT
#define NESO_PARTICLES_PETSC_VERSION_LT(x, y)                                  \
  ((PETSC_VERSION_MAJOR < (x)) ||                                              \
   ((PETSC_VERSION_MAJOR == (x)) && (PETSC_VERSION_MINOR < (y))))
#endif

namespace NESO::Particles::NPPETScAPI {

/**
 * Wraps DMPlexCreateBoxMesh to work pre and post petsc v3.22.
 *
 * @ingroup external_interfaces_petsc_dmplex_helper_functions
 * @param[in] comm Communicator for mesh.
 * @param[in] dim Number of spatial dimensions.
 * @param[in] simplex Create a mesh consisting of triangles.
 * @param[in] faces Number of faces in each dimension.
 * @param[in] lower Origin for each dimension.
 * @param[in] upper Extent plus origin for each dimension.
 * @param[in] periodicity Periodicity of each dimension.
 * @param[in] interpolate Should PETSc interpolate the mesh.
 * @param[in, out] dm Output DM.
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

/**
 * Wraps PetscErrorMessage to work pre and post petsc v3.25.
 *
 * @param error_code PETSc error code to get messages for.
 * @returns PETSc text and specific as strings.
 */
inline std::pair<std::string, std::string>
NP_PetscErrorMessage(const PetscInt error_code) {

#if NESO_PARTICLES_PETSC_VERSION_LT(3, 25)
  const char *text;
  char *specific;
  PetscErrorMessage(error_code, &text, &specific);
  return {text, specific};
#else
  const char *text;
  const char *specific;
  PetscErrorMessage(error_code, &text, &specific);
  return {text, specific};
#endif
}

}; // namespace NESO::Particles::NPPETScAPI

#endif
