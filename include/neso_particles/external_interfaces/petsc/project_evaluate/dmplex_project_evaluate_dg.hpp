#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_DG_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_DG_H_

#include "dmplex_project_evaluate_base.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Implementation to deposit particle data into and evalute from DG0 function
 * spaces. This implementation has an internal state (cdc_project) which is a
 * CellDatConst that holds the DOFs. This cell dat is resized whenever set_dofs
 * is called with a greater number of components or when project is called with
 * a greater number of components. Calling project or set_dofs will overwrite
 * the existing DOFs. Calling get_dofs or evaluate will use or retrieve the
 * existing DOFs.
 */
class DMPlexProjectEvaluateDG : public DMPlexProjectEvaluateBase {
protected:
  std::shared_ptr<CellDatConst<REAL>> cdc_project;
  std::shared_ptr<CellDatConst<REAL>> cdc_volumes;
  DMPlexInterfaceSharedPtr mesh;
  SYCLTargetSharedPtr sycl_target;

  void check_setup();
  void check_ncomp(const int ncomp);

  template <typename T>
  inline void project_inner(std::shared_ptr<T> particle_sub_group,
                            Sym<REAL> sym) {
    auto particle_group = get_particle_group(particle_sub_group);
    auto dat = particle_group->get_dat(sym);
    const int ncomp = dat->ncomp;
    this->check_ncomp(ncomp);

    // DG0 projection onto the CellDatConst
    this->cdc_project->fill(0.0);
    particle_loop(
        "DMPlexProjectEvaluateDG::project_0", particle_sub_group,
        [=](auto SRC, auto DST, auto VOLS) {
          const REAL iv = VOLS.at(0, 0);
          for (int cx = 0; cx < ncomp; cx++) {
            DST.combine(cx, 0, SRC.at(cx) * iv);
          }
        },
        Access::read(sym),
        Access::reduce(this->cdc_project, Kernel::plus<REAL>()),
        Access::read(this->cdc_volumes))
        ->execute();

    if (this->qpm != nullptr) {
      // Read the CellDatConst values onto the quadrature point values
      auto destination_dat = this->qpm->get_sym(ncomp);
      particle_loop(
          "DMPlexProjectEvaluateDG::project_1", this->qpm->particle_group,
          [=](auto SRC, auto DST) {
            for (int cx = 0; cx < ncomp; cx++) {
              DST.at(cx) = SRC.at(cx, 0);
            }
          },
          Access::read(this->cdc_project), Access::write(destination_dat))
          ->execute();
    }
  }

  template <typename T>
  inline void evaluate_inner(std::shared_ptr<T> particle_sub_group,
                             Sym<REAL> sym) {

    auto particle_group = get_particle_group(particle_sub_group);
    auto dat = particle_group->get_dat(sym);
    const int ncomp = dat->ncomp;
    this->check_ncomp(ncomp);

    if (this->qpm != nullptr) {
      // Copy from the quadrature point values into the CellDatConst
      auto source_dat = this->qpm->get_sym(ncomp);
      particle_loop(
          "DMPlexProjectEvaluateDG::evaluate_0", this->qpm->particle_group,
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
    }

    // Copy from the CellDatConst to the output particle group
    particle_loop(
        "DMPlexProjectEvaluateDG::evaluate_1", particle_sub_group,
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
  virtual ~DMPlexProjectEvaluateDG() = default;
  DMPlexProjectEvaluateDG() = default;

  /**
   * Get a representation of the internal state which can be passed to the
   * VTKHDF writer.
   *
   * @returns Data for VTKHDF unstructured grid writer.
   */
  virtual std::vector<VTK::UnstructuredCell> get_vtk_data() override;

  /**
   * Create a DG0 project/evaluate instance from a QuadraturePointMapper.
   *
   * @param qpm QuadraturePointMapper with a single point per cell.
   * @param function_space Should be "DG".
   * @param polynomial_order Should be 0.
   */
  DMPlexProjectEvaluateDG(ExternalCommon::QuadraturePointMapperSharedPtr qpm,
                          std::string function_space, int polynomial_order);

  /**
   * Create a DG project/evaluate instance on a mesh. This constructor will
   * create an instance that assumes that the MPI rank that owns a cell also
   * owns all DOFs associated with that cell. i.e. there is no
   * QuadraturePointMapper moving values around.
   *
   * @param mesh DMPlexInterface mesh object to use as domain for functions.
   * @param sycl_target Compute device that will be used.
   * @param function_space Function space to use, e.g. "DG".
   * @param polynomial_order Polynomial order to use, e.g. 0.
   */
  DMPlexProjectEvaluateDG(DMPlexInterfaceSharedPtr mesh,
                          SYCLTargetSharedPtr sycl_target,
                          std::string function_space, int polynomial_order);

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
  virtual void project(ParticleGroupSharedPtr particle_group,
                       Sym<REAL> sym) override;

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
  virtual void project(ParticleSubGroupSharedPtr particle_sub_group,
                       Sym<REAL> sym) override;

  /**
   * If the output sym has ncomp components then this method evaluates the
   * function defined in the point properties given by
   * QuadraturePointMapper::get_sym(ncomp) at the location of each particle.
   *
   * @param particle_group Set of particles to evaluate the deposited function
   * at.
   * @param sym Output particle property on which to place evaluations.
   */
  virtual void evaluate(ParticleGroupSharedPtr particle_group,
                        Sym<REAL> sym) override;

  /**
   * If the output sym has ncomp components then this method evaluates the
   * function defined in the point properties given by
   * QuadraturePointMapper::get_sym(ncomp) at the location of each particle.
   *
   * @param particle_sub_group Set of particles to evaluate the deposited
   * function at.
   * @param sym Output particle property on which to place evaluations.
   */
  virtual void evaluate(ParticleSubGroupSharedPtr particle_sub_group,
                        Sym<REAL> sym) override;

  /**
   * Get the DOFs for all cells and all components from the last project or
   * set_dofs call. DOF ordering is the per cell components running fastest then
   * cells.
   *
   * @param ncomp[in] Number of components/functions to retrieve DOFs for. e.g.
   * if project has been called for a particle property with two components then
   * there are two DOFs per cell in the internal representation.
   * @param[in, out] DOFs in std::vector<REAL> form. This array may be resized
   * and zeroed by the implementation.
   */
  void get_dofs(const int ncomp, std::vector<REAL> &dofs);

  /**
   * Set the DOFs for the internal representation, e.g. prior to an evaluate
   * call. DOF ordering is the per cell components running fastest then
   * cells.
   *
   * @param ncomp[in] Number of components/functions to retrieve DOFs for. e.g.
   * if project has been called for a particle property with two components then
   * there are two DOFs per cell in the internal representation.
   * @param[in] DOFs in std::vector<REAL> form.
   */
  void set_dofs(const int ncomp, const std::vector<REAL> &dofs);
};

} // namespace NESO::Particles::PetscInterface

#endif
