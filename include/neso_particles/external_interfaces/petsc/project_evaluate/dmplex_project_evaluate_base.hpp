#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BASE_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BASE_H_

#include "../../common/quadrature_point_mapper.hpp"
#include "../../petsc/dmplex_interface.hpp"
#include "../../vtk/vtk.hpp"

namespace NESO::Particles::PetscInterface {

class DMPlexProjectEvaluateBase {
protected:
public:
  /**
   * TODO
   * Projects values from particle data onto the values in the
   * QuadraturePointMapper.
   */
  virtual inline void project(ParticleGroupSharedPtr particle_group,
                              Sym<REAL> sym) = 0;

  /**
   * TODO
   * Evaluates the function defined by the values in the QuadraturePointValues
   * at the particle locations.
   */
  virtual inline void evaluate(ParticleGroupSharedPtr particle_group,
                               Sym<REAL> sym) = 0;

  /**
   * TODO
   */
  virtual inline std::vector<VTK::UnstructuredCell> get_vtk_data() = 0;
};

} // namespace NESO::Particles::PetscInterface

#endif
