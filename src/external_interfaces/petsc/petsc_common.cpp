#ifdef NESO_PARTICLES_PETSC
#include <neso_particles/external_interfaces/petsc/petsc_common.hpp>

namespace NESO::Particles::PetscInterface {

template void neso_particles_petsc_error<PetscInt>(const char *expr_str,
                                                   PetscInt error_code,
                                                   const char *file, int line);
}

#endif
