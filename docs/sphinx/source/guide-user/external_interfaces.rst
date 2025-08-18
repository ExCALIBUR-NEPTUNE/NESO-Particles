*******************
External Interfaces
*******************

Header Files
============

The header files for the external interfaces are separate from the main `neso_particles.hpp` header file to improve compilation times.

::

  // For the PETSc interface.
  #include <neso_particles/external_interfaces/petsc/petsc_interface.hpp>

  // For the common utilities.
  #include "neso_particles/external_interfaces/common/common.hpp"

  // For the built in vtk implementation.
  #include "neso_particles/external_interfaces/vtk/vtk.hpp"


