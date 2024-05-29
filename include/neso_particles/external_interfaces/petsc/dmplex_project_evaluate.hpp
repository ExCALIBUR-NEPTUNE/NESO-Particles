#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_H_

#include "../common/quadrature_point_mapper.hpp"
#include "../petsc/dmplex_interface.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * TODO
 */
class DMPlexProjectEvaluate {
protected:
  std::shared_ptr<CellDatConst<REAL>> cdc_project;
  std::shared_ptr<CellDatConst<REAL>> cdc_volumes;
  DMPlexInterfaceSharedPtr mesh;

  inline void check_setup() {
    NESOASSERT(this->qpm->points_added(),
               "QuadraturePointMapper needs points adding to it.");
  }

  inline void project_dg_0(ParticleGroupSharedPtr particle_group,
                           Sym<REAL> sym) {
    auto dat = particle_group->get_dat(sym);
    const int ncomp = dat->ncomp;
    auto destination_dat = this->qpm->get_sym(ncomp);

    // DG0 projection onto the CellDatConst
    this->cdc_project->fill(0.0);
    particle_loop(
        "DMPlexProjectEvaluate::project_0", particle_group,
        [=](auto SRC, auto DST) {
          for (int cx = 0; cx < ncomp; cx++) {
            DST.fetch_add(cx, 0, SRC.at(cx));
          }
        },
        Access::read(sym), Access::add(this->cdc_project))
        ->execute();

    // Read the CellDatConst values onto the quadrature point values
    particle_loop(
        "DMPlexProjectEvaluate::project_1", this->qpm->particle_group,
        [=](auto VOLS, auto SRC, auto DST) {
          const REAL iv = 1.0 / VOLS.at(0, 0);
          for (int cx = 0; cx < ncomp; cx++) {
            DST.at(cx) = SRC.at(cx, 0) * iv;
          }
        },
        Access::read(this->cdc_volumes), Access::read(this->cdc_project),
        Access::write(destination_dat))
        ->execute();
  }

  inline void evaluate_dg_0(ParticleGroupSharedPtr particle_group,
                            Sym<REAL> sym) {
    auto dat = particle_group->get_dat(sym);
    const int ncomp = dat->ncomp;
    auto source_dat = this->qpm->get_sym(ncomp);

    // Copy from the quadrature point values into the CellDatConst
    particle_loop(
        "DMPlexProjectEvaluate::evaluate_0", this->qpm->particle_group,
        [=](auto SRC, auto DST, auto MASK) {
          if (MASK.at(3) == 0) {
            for (int cx = 0; cx < ncomp; cx++) {
              DST.at(cx, 0) = SRC.at(cx);
            }
          }
        },
        Access::read(source_dat), Access::write(this->cdc_project),
        Access::read(Sym<INT>("ADDING_RANK_INDEX")))
        ->execute();

    // Copy from the CellDatConst to the output particle group
    particle_loop(
        "DMPlexProjectEvaluate::evaluate_1", particle_group,
        [=](auto SRC, auto DST) {
          for (int cx = 0; cx < ncomp; cx++) {
            DST.at(cx) = SRC.at(cx, 0);
          }
        },
        Access::read(this->cdc_project), Access::write(sym))
        ->execute();
  }

public:
  ExternalCommon::QuadraturePointMapperSharedPtr qpm;
  std::string function_space;
  int polynomial_order;
  DMPlexProjectEvaluate() = default;

  /**
   * TODO
   */
  DMPlexProjectEvaluate(ExternalCommon::QuadraturePointMapperSharedPtr qpm,
                        std::string function_space, int polynomial_order)
      : qpm(qpm), function_space(function_space),
        polynomial_order(polynomial_order) {
    NESOASSERT(this->qpm != nullptr, "QuadraturePointMapper is nullptr");
    NESOASSERT(function_space == "DG",
               "Only DG function spaces currently supported");
    NESOASSERT(polynomial_order == 0,
               "Only order 0 function spaces currently supported");

    this->mesh = std::dynamic_pointer_cast<PetscInterface::DMPlexInterface>(
        this->qpm->domain->mesh);
    NESOASSERT(this->mesh != nullptr,
               "Mesh is not descendent from PetscInterface::DMPlexInterface");
    NESOASSERT(this->mesh->get_ndim() == 2, "Only implemented for 2D domains.");

    const int cell_count = this->qpm->domain->mesh->get_cell_count();
    this->cdc_project = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 1, 1);
    this->cdc_volumes = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 1, 1);

    // For each cell record the volume.
    for (int cx = 0; cx < cell_count; cx++) {
      const auto volume = this->mesh->dmh->get_cell_volume(cx);
      this->cdc_volumes->set_value(cx, 0, 0, volume);
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
    this->project_dg_0(particle_group, sym);
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
    this->evaluate_dg_0(particle_group, sym);
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
