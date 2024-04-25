#ifndef _NESO_PARTICLES_PETSC_COMMON_HPP_
#define _NESO_PARTICLES_PETSC_COMMON_HPP_

#include "../../communication.hpp"
#include "../../typedefs.hpp"
#include "petscdm.h"
#include "petscdmlabel.h"
#include "petscds.h"
#include <petscdmplex.h>
#include <petscerror.h>
#include <petscsf.h>

namespace NESO::Particles::PetscInterface {

/**
 * PETSc: doc/changes/319.rst:
 * Add ``PETSC_SUCCESS`` to indicate success, always guaranteed to equal ``0``.
 */
#ifdef PETSC_SUCCESS
#define NESO_PARTICLES_PETSC_SUCCESS PETSC_SUCCESS
#else
#define NESO_PARTICLES_PETSC_SUCCESS 0
#endif

/**
 * This is a helper function to assert conditions are satisfied and terminate
 * execution if not.
 *
 *   PETSCCHK(<PETSc call>);
 *
 * To check conditionals within their code.
 *
 * @param expr_str A string identifying the conditional to check.
 * @param error_code Error code from PETSc call.
 * @param file Filename containing the call to neso_particles_assert.
 * @param line Line number for the call to neso_particles assert.
 */
template <typename T>
inline void neso_particles_petsc_error(const char *expr_str, T error_code,
                                       const char *file, int line) {
  if (error_code != NESO_PARTICLES_PETSC_SUCCESS) {
    const char *text;
    char *specific;
    PetscErrorMessage(error_code, &text, &specific);
    nprint("Error Code:", error_code, "\n", text, "\n", specific);
    neso_particles_assert(expr_str, false, file, line,
                          "PETSc call did not return PETSC_SUCCESS");
  }
}

} // namespace NESO::Particles::PetscInterface

/**
 * \def PETSCCHK(expr, msg)
 * This is a helper macro to call the function neso_particles_petsc_error. Users
 * should call this helper macro PETSCCHK like
 *
 *   PETSCCHK(<PETSc call>);
 *
 * To check PETSc calls within their code.
 */
#define PETSCCHK(expr)                                                         \
  NESO::Particles::PetscInterface::neso_particles_petsc_error(                 \
      #expr, expr, __FILE__, __LINE__)

#endif
