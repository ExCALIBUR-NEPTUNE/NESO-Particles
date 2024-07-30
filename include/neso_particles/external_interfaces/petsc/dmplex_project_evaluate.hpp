#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_H_

#include "project_evaluate/dmplex_project_evaluate_barycentric.hpp"
#include "project_evaluate/dmplex_project_evaluate_base.hpp"
#include "project_evaluate/dmplex_project_evaluate_dg.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Type to handle projection/deposition onto and evaluation from a set of
 * quadrature (nodal) points.
 */
class DMPlexProjectEvaluate {
protected:
  inline void check_setup() {
    NESOASSERT(this->implementation != nullptr,
               "No implementation was created.");
    NESOASSERT(this->qpm->points_added(),
               "QuadraturePointMapper needs points adding to it.");
  }

public:
  DMPlexInterfaceSharedPtr mesh;
  ExternalCommon::QuadraturePointMapperSharedPtr qpm;
  std::string function_space;
  int polynomial_order;
  std::shared_ptr<DMPlexProjectEvaluateBase> implementation;
  DMPlexProjectEvaluate() = default;

  /**
   * Create a handler for projection/deposition onto and evaluation from a set
   * of quadrature (nodal) points. The currently implemented function spaces
   * and polynomial orders are:
   *
   *    function_space="DG", polynomial_order=0
   *    function_space="Barycentric", polynomial_order=1
   *
   * @param qpm QuadraturePointMapper which describes the quadrature points on
   * which to deposit and from which functions are evaluated.
   * @param function_space String which specifies the type of deposition and
   * evaluation.
   * @param polynomial_order Polynomial order to use with the specified function
   * space.
   */
  DMPlexProjectEvaluate(ExternalCommon::QuadraturePointMapperSharedPtr qpm,
                        std::string function_space, int polynomial_order)
      : qpm(qpm), function_space(function_space),
        polynomial_order(polynomial_order) {

    std::map<std::string, std::pair<int, int>> map_allowed;
    map_allowed["DG"] = {0, 0};
    map_allowed["Barycentric"] = {1, 1};

    NESOASSERT(this->qpm != nullptr, "QuadraturePointMapper is nullptr");
    NESOASSERT(map_allowed.count(function_space),
               "Only function space: " + function_space + " not recognised.");
    const int p_min = map_allowed.at(function_space).first;
    const int p_max = map_allowed.at(function_space).second;
    NESOASSERT(((p_min <= polynomial_order) && (polynomial_order <= p_max)),
               "Polynomial order " + std::to_string(polynomial_order) +
                   " outside of acceptable range [" + std::to_string(p_min) +
                   ", " + std::to_string(p_max) + "] for function space " +
                   function_space + ".");

    this->mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
        this->qpm->domain->mesh);
    NESOASSERT(this->mesh != nullptr,
               "Mesh is not descendent from PetscInterface::DMPlexInterface");

    if (function_space == "DG") {
      this->implementation =
          std::dynamic_pointer_cast<DMPlexProjectEvaluateBase>(
              std::make_shared<DMPlexProjectEvaluateDG>(qpm, function_space,
                                                        polynomial_order));
    } else if (function_space == "Barycentric") {
      this->implementation =
          std::dynamic_pointer_cast<DMPlexProjectEvaluateBase>(
              std::make_shared<DMPlexProjectEvaluateBarycentric>(
                  qpm, function_space, polynomial_order));
    }
  }

  /**
   * Get a representation of the internal state which can be passed to the
   * VTKHDF writer. This method returns a representation that corresponds to
   * the last projection or evaluation which occured with this instance.
   *
   * @returns Data for VTKHDF unstructured grid writer.
   */
  inline std::vector<VTK::UnstructuredCell> get_vtk_data() {
    return this->implementation->get_vtk_data();
  }

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
  inline void project(ParticleGroupSharedPtr particle_group, Sym<REAL> sym) {
    this->check_setup();
    NESOASSERT(particle_group->contains_dat(sym),
               "ParticleGroup does not contain the sym: " + sym.name);
    this->implementation->project(particle_group, sym);
  }

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
  inline void project(ParticleSubGroupSharedPtr particle_sub_group,
                      Sym<REAL> sym) {
    this->check_setup();
    auto particle_group = get_particle_group(particle_sub_group);
    NESOASSERT(particle_group->contains_dat(sym),
               "ParticleGroup does not contain the sym: " + sym.name);
    this->implementation->project(particle_sub_group, sym);
  }

  /**
   * If the output sym has ncomp components then this method evaluates the
   * function defined in the point properties given by
   * QuadraturePointMapper::get_sym(ncomp) at the location of each particle.
   *
   * @param particle_group Set of particles to evaluate the deposited function
   * at.
   * @param sym Output particle property on which to place evaluations.
   */
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<REAL> sym) {
    this->check_setup();
    NESOASSERT(particle_group->contains_dat(sym),
               "ParticleGroup does not contain the sym: " + sym.name);
    this->implementation->evaluate(particle_group, sym);
  }

  /**
   * If the output sym has ncomp components then this method evaluates the
   * function defined in the point properties given by
   * QuadraturePointMapper::get_sym(ncomp) at the location of each particle.
   *
   * @param particle_sub_group Set of particles to evaluate the deposited
   * function at.
   * @param sym Output particle property on which to place evaluations.
   */
  inline void evaluate(ParticleSubGroupSharedPtr particle_sub_group,
                       Sym<REAL> sym) {
    this->check_setup();
    auto particle_group = get_particle_group(particle_sub_group);
    NESOASSERT(particle_group->contains_dat(sym),
               "ParticleGroup does not contain the sym: " + sym.name);
    this->implementation->evaluate(particle_sub_group, sym);
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
