#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_DG_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_DG_H_

#include "dmplex_project_evaluate_base.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Implementation to deposit particle data into and evalute from DG0 function
 * spaces.
 */
class DMPlexProjectEvaluateDG : public DMPlexProjectEvaluateBase {
protected:
  std::shared_ptr<CellDatConst<REAL>> cdc_project;
  std::shared_ptr<CellDatConst<REAL>> cdc_volumes;
  DMPlexInterfaceSharedPtr mesh;

  inline void check_setup() {
    NESOASSERT(this->qpm->points_added(),
               "QuadraturePointMapper needs points adding to it.");
  }

  inline void check_ncomp(const int ncomp) {
    if (this->cdc_project->nrow < ncomp) {
      const int cell_count = this->qpm->domain->mesh->get_cell_count();
      this->cdc_project = std::make_shared<CellDatConst<REAL>>(
          this->qpm->sycl_target, cell_count, ncomp, 1);
    }
  }

  template <typename T>
  inline void project_inner(std::shared_ptr<T> particle_sub_group,
                            Sym<REAL> sym) {
    auto particle_group = get_particle_group(particle_sub_group);
    auto dat = particle_group->get_dat(sym);
    const int ncomp = dat->ncomp;
    this->check_ncomp(ncomp);
    auto destination_dat = this->qpm->get_sym(ncomp);

    // DG0 projection onto the CellDatConst
    this->cdc_project->fill(0.0);
    particle_loop(
        "DMPlexProjectEvaluateDG::project_0", particle_sub_group,
        [=](auto SRC, auto DST) {
          for (int cx = 0; cx < ncomp; cx++) {
            DST.fetch_add(cx, 0, SRC.at(cx));
          }
        },
        Access::read(sym), Access::add(this->cdc_project))
        ->execute();

    // Read the CellDatConst values onto the quadrature point values
    particle_loop(
        "DMPlexProjectEvaluateDG::project_1", this->qpm->particle_group,
        [=](auto VOLS, auto SRC, auto DST) {
          const REAL iv = VOLS.at(0, 0);
          for (int cx = 0; cx < ncomp; cx++) {
            DST.at(cx) = SRC.at(cx, 0) * iv;
          }
        },
        Access::read(this->cdc_volumes), Access::read(this->cdc_project),
        Access::write(destination_dat))
        ->execute();
  }

  template <typename T>
  inline void evaluate_inner(std::shared_ptr<T> particle_sub_group,
                             Sym<REAL> sym) {

    auto particle_group = get_particle_group(particle_sub_group);
    auto dat = particle_group->get_dat(sym);
    const int ncomp = dat->ncomp;
    this->check_ncomp(ncomp);
    auto source_dat = this->qpm->get_sym(ncomp);

    // Copy from the quadrature point values into the CellDatConst
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
  DMPlexProjectEvaluateDG() = default;

  /**
   * Get a representation of the internal state which can be passed to the
   * VTKHDF writer.
   *
   * @returns Data for VTKHDF unstructured grid writer.
   */
  virtual inline std::vector<VTK::UnstructuredCell> get_vtk_data() override {
    const int cell_count = this->mesh->get_cell_count();
    std::vector<VTK::UnstructuredCell> data(cell_count);
    std::vector<std::vector<REAL>> vertices;
    const int ndim = mesh->get_ndim();
    for (int cellx = 0; cellx < cell_count; cellx++) {
      vertices.clear();
      mesh->dmh->get_cell_vertices(cellx, vertices);
      const int num_vertices = vertices.size();
      auto inverse_volumes = this->cdc_volumes->get_value(cellx, 0, 0);
      const auto cell_value = this->cdc_project->get_value(cellx, 0, 0);
      data.at(cellx).num_points = num_vertices;
      data.at(cellx).num_dimensions = 2;
      data.at(cellx).points.reserve(num_vertices * 3);
      for (int vx = 0; vx < num_vertices; vx++) {
        for (int dx = 0; dx < ndim; dx++) {
          data.at(cellx).points.push_back(vertices.at(vx).at(dx));
        }
        for (int dx = ndim; dx < 3; dx++) {
          data.at(cellx).points.push_back(0.0);
        }
        data.at(cellx).cell_data["value"] = cell_value * inverse_volumes;
      }
    }
    return data;
  }

  /**
   * Create a DG0 project/evaluate instance from a QuadraturePointMapper.
   *
   * @param qpm QuadraturePointMapper with a single point per cell.
   * @param function_space Should be "DG".
   * @param polynomial_order Should be 0.
   */
  DMPlexProjectEvaluateDG(ExternalCommon::QuadraturePointMapperSharedPtr qpm,
                          std::string function_space, int polynomial_order)
      : qpm(qpm), function_space(function_space),
        polynomial_order(polynomial_order) {

    std::map<std::string, std::pair<int, int>> map_allowed;
    map_allowed["DG"] = {0, 0};

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
    NESOASSERT(this->mesh->get_ndim() == 2, "Only implemented for 2D domains.");

    const int cell_count = this->qpm->domain->mesh->get_cell_count();
    this->cdc_project = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 1, 1);
    this->cdc_volumes = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 1, 1);

    // For each cell record the volume.
    for (int cx = 0; cx < cell_count; cx++) {
      const auto volume = this->mesh->dmh->get_cell_volume(cx);
      this->cdc_volumes->set_value(cx, 0, 0, 1.0 / volume);
    }
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
  virtual inline void project(ParticleGroupSharedPtr particle_group,
                              Sym<REAL> sym) override {
    this->project_inner(particle_group, sym);
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
  virtual inline void project(ParticleSubGroupSharedPtr particle_sub_group,
                              Sym<REAL> sym) override {
    this->project_inner(particle_sub_group, sym);
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
  virtual inline void evaluate(ParticleGroupSharedPtr particle_group,
                               Sym<REAL> sym) override {
    this->evaluate_inner(particle_group, sym);
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
  virtual inline void evaluate(ParticleSubGroupSharedPtr particle_sub_group,
                               Sym<REAL> sym) override {
    this->evaluate_inner(particle_sub_group, sym);
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
