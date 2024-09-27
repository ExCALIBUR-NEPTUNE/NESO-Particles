#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BASE_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BASE_H_

#include "../../common/quadrature_point_mapper.hpp"
#include "../../petsc/dmplex_interface.hpp"
#include "../../vtk/vtk.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Abstract base class for Projection/Evaluation implementations.
 */
class DMPlexProjectEvaluateBase {
protected:
public:
  /**
   * Projects values from particle data onto the values in the
   * QuadraturePointMapper.
   *
   * @param particle_group ParticleGroup of particles containing data to deposit
   * onto the grid.
   * @param sym Sym objects which indices which particle data to deposit. This
   * projects into the QuadraturePointMapper particle group into the particle
   * properties given by calling QuadraturePointMapper::get_sym(ncomp).
   */
  virtual inline void project(ParticleGroupSharedPtr particle_group,
                              Sym<REAL> sym) = 0;

  /**
   * Projects values from particle data onto the values in the
   * QuadraturePointMapper.
   *
   * @param particle_sub_group ParticleSubGroup of particles containing data to
   * deposit onto the grid.
   * @param sym Sym objects which indices which particle data to deposit. This
   * projects into the QuadraturePointMapper particle group into the particle
   * properties given by calling QuadraturePointMapper::get_sym(ncomp).
   */
  virtual inline void project(ParticleSubGroupSharedPtr particle_sub_group,
                              Sym<REAL> sym) = 0;

  /**
   * If the output sym has ncomp components then this method evaluates the
   * function defined in the point properties given by
   * QuadraturePointMapper::get_sym(ncomp) at the location of each particle.
   *
   * @param particle_group Set of particles to evaluate the deposited function
   * at.
   * @param sym Output particle property on which to place evaluations.
   */
  virtual inline void evaluate(ParticleGroupSharedPtr particle_group,
                               Sym<REAL> sym) = 0;

  /**
   * If the output sym has ncomp components then this method evaluates the
   * function defined in the point properties given by
   * QuadraturePointMapper::get_sym(ncomp) at the location of each particle.
   *
   * @param particle_sub_group Set of particles to evaluate the deposited
   * function at.
   * @param sym Output particle property on which to place evaluations.
   */
  virtual inline void evaluate(ParticleSubGroupSharedPtr particle_sub_group,
                               Sym<REAL> sym) = 0;

  /**
   * Get a representation of the internal state which can be passed to the
   * VTKHDF writer.
   *
   * @returns Data for VTKHDF unstructured grid writer.
   */
  virtual inline std::vector<VTK::UnstructuredCell> get_vtk_data() = 0;
};

} // namespace NESO::Particles::PetscInterface

#endif
