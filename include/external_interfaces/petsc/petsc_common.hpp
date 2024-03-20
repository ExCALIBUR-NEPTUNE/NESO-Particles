#ifndef _NESO_PARTICLES_PETSC_COMMON_HPP_
#define _NESO_PARTICLES_PETSC_COMMON_HPP_

#include "../../communication.hpp"
#include "../../typedefs.hpp"
#include <petscdmplex.h>
#include <petscerror.h>
#include <petscsf.h>

#define PETSCCHK(cmd) NESOASSERT((cmd) == PETSC_SUCCESS, "PETSC ERROR");
#endif
