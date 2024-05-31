#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_H_

#include "project_evaluate/dmplex_project_evaluate_barycentric.hpp"
#include "project_evaluate/dmplex_project_evaluate_base.hpp"
#include "project_evaluate/dmplex_project_evaluate_dg.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * TODO
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
   * TODO
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
      std::dynamic_pointer_cast<DMPlexProjectEvaluateBase>(
          std::make_shared<DMPlexProjectEvaluateBarycentric>(
              qpm, function_space, polynomial_order));
    }
  }

  /**
   * TODO
   * Projects values from particle data onto the values in the
   * QuadraturePointMapper.
   */
  inline void project(ParticleGroupSharedPtr particle_group, Sym<REAL> sym) {
    this->check_setup();
    NESOASSERT(particle_group->contains_dat(sym),
               "ParticleGroup does not contain the sym: " + sym.name);
    this->implementation->project(particle_group, sym);
  }

  /**
   * TODO
   * Evaluates the function defined by the values in the QuadraturePointValues
   * at the particle locations.
   */
  inline void evaluate(ParticleGroupSharedPtr particle_group, Sym<REAL> sym) {
    this->check_setup();
    NESOASSERT(particle_group->contains_dat(sym),
               "ParticleGroup does not contain the sym: " + sym.name);
    this->implementation->evaluate(particle_group, sym);
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
