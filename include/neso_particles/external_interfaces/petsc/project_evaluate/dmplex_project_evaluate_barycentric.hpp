#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BARYCENTRIC_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BARYCENTRIC_H_

#include "dmplex_project_evaluate_base.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Convert 2D Cartesian coordinates to Barycentric coordinates.
 *
 *  @param[in] x1 Triangle vertex 1, x component.
 *  @param[in] y1 Triangle vertex 1, y component.
 *  @param[in] x2 Triangle vertex 2, x component.
 *  @param[in] y2 Triangle vertex 2, y component.
 *  @param[in] x3 Triangle vertex 3, x component.
 *  @param[in] y3 Triangle vertex 3, y component.
 *  @param[in] x Point Cartesian coordinate, x component.
 *  @param[in] y Point Cartesian coordinate, y component.
 *  @param[in, out] l1 Point Barycentric coordinate, lambda 1.
 *  @param[in, out] l2 Point Barycentric coordinate, lambda 2.
 *  @param[in, out] l3 Point Barycentric coordinate, lambda 3.
 */
inline void
triangle_cartesian_to_barycentric(const REAL x1, const REAL y1, const REAL x2,
                                  const REAL y2, const REAL x3, const REAL y3,
                                  const REAL x, const REAL y, REAL *RESTRICT l1,
                                  REAL *RESTRICT l2, REAL *RESTRICT l3) {
  const REAL scaling = 1.0 / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
  *l1 = scaling * ((x2 * y3 - x3 * y2) + (y2 - y3) * x + (x3 - x2) * y);
  *l2 = scaling * ((x3 * y1 - x1 * y3) + (y3 - y1) * x + (x1 - x3) * y);
  *l3 = scaling * ((x1 * y2 - x2 * y1) + (y1 - y2) * x + (x2 - x1) * y);
};

/**
 * Convert 2D Barycentric coordinates to Cartesian coordinates.
 *
 *  @param[in] x1 Triangle vertex 1, x component.
 *  @param[in] y1 Triangle vertex 1, y component.
 *  @param[in] x2 Triangle vertex 2, x component.
 *  @param[in] y2 Triangle vertex 2, y component.
 *  @param[in] x3 Triangle vertex 3, x component.
 *  @param[in] y3 Triangle vertex 3, y component.
 *  @param[in] l1 Point Barycentric coordinate, lambda 1.
 *  @param[in] l2 Point Barycentric coordinate, lambda 2.
 *  @param[in] l3 Point Barycentric coordinate, lambda 3.
 *  @param[in, out] x Point Cartesian coordinate, x component.
 *  @param[in, out] y Point Cartesian coordinate, y component.
 */
inline void triangle_barycentric_to_cartesian(const REAL x1, const REAL y1,
                                              const REAL x2, const REAL y2,
                                              const REAL x3, const REAL y3,
                                              const REAL l1, const REAL l2,
                                              const REAL l3, REAL *RESTRICT x,
                                              REAL *RESTRICT y) {
  *x = l1 * x1 + l2 * x2 + l3 * x3;
  *y = l1 * y1 + l2 * y2 + l3 * y3;
};

/**
 * TODO
 */
class DMPlexProjectEvaluateBarycentric : public DMPlexProjectEvaluateBase {
protected:
  std::shared_ptr<CellDatConst<REAL>> cdc_project;
  std::shared_ptr<CellDatConst<int>> cdc_num_vertices;
  std::shared_ptr<CellDatConst<REAL>> cdc_vertices;
  DMPlexInterfaceSharedPtr mesh;

  inline void check_setup() {
    NESOASSERT(this->qpm->points_added(),
               "QuadraturePointMapper needs points adding to it.");
  }

public:
  ExternalCommon::QuadraturePointMapperSharedPtr qpm;
  std::string function_space;
  int polynomial_order;
  DMPlexProjectEvaluateBarycentric() = default;

  /**
   * TODO
   */
  DMPlexProjectEvaluateBarycentric(
      ExternalCommon::QuadraturePointMapperSharedPtr qpm,
      std::string function_space, int polynomial_order)
      : qpm(qpm), function_space(function_space),
        polynomial_order(polynomial_order) {

    std::map<std::string, std::pair<int, int>> map_allowed;
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
    NESOASSERT(this->mesh->get_ndim() == 2, "Only implemented for 2D domains.");

    const int cell_count = this->qpm->domain->mesh->get_cell_count();
    this->cdc_project = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 4, 1);
    this->cdc_num_vertices = std::make_shared<CellDatConst<int>>(
        this->qpm->sycl_target, cell_count, 1, 1);
    this->cdc_vertices = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 4, 2);

    // For each cell record the volume.
    std::vector<std::vector<REAL>> vertices;
    for (int cx = 0; cx < cell_count; cx++) {
      this->mesh->dmh->get_cell_vertices(cx, vertices);
      const int num_verts = vertices.size();
      // TODO implement for quads
      NESOASSERT(num_verts < 4, "Unexpected number of vertices (expected 3).");
      this->cdc_num_vertices->set_value(cx, 0, 0, 3);
      for (int vx = 0; vx < 3; vx++) {
        for (int dx = 0; dx < 2; dx++) {
          this->cdc_vertices->set_value(cx, vx, dx, vertices.at(vx).at(dx));
        }
      }
    }
  }

  /**
   * TODO
   * Projects values from particle data onto the values in the
   * QuadraturePointMapper.
   */
  inline void project(ParticleGroupSharedPtr particle_group,
                      Sym<REAL> sym) override {
    auto dat = particle_group->get_dat(sym);
    auto position_dat = particle_group->position_dat;

    const int ncomp = dat->ncomp;
    auto destination_dat = this->qpm->get_sym(ncomp);

    // DG0 projection onto the CellDatConst
    this->cdc_project->fill(0.0);
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::project_0", particle_group,
        [=](auto NUM_VERTICES, auto VERTICES, auto SRC, auto DST) {
          const int num_vertices = NUM_VERTICES.at(0, 0);
        },
        Access::read(this->cdc_num_vertices), Access::read(this->cdc_vertices),
        Access::read(sym), Access::add(this->cdc_project))
        ->execute();

    // Read the CellDatConst values onto the quadrature point values
    // particle_loop(
    //    "DMPlexProjectEvaluateBarycentric::project_1",
    //    this->qpm->particle_group,
    //    [=](auto VOLS, auto SRC, auto DST) {
    //      const REAL iv = 1.0 / VOLS.at(0, 0);
    //      for (int cx = 0; cx < ncomp; cx++) {
    //        DST.at(cx) = SRC.at(cx, 0) * iv;
    //      }
    //    },
    //    Access::read(this->cdc_project),
    //    Access::write(destination_dat))
    //    ->execute();
  }

  /**
   * TODO
   * Evaluates the function defined by the values in the QuadraturePointValues
   * at the particle locations.
   */
  inline void evaluate(ParticleGroupSharedPtr particle_group,
                       Sym<REAL> sym) override {
    auto dat = particle_group->get_dat(sym);
    const int ncomp = dat->ncomp;
    auto source_dat = this->qpm->get_sym(ncomp);

    // Copy from the quadrature point values into the CellDatConst
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::evaluate_0",
        this->qpm->particle_group,
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
        "DMPlexProjectEvaluateBarycentric::evaluate_1", particle_group,
        [=](auto SRC, auto DST) {
          for (int cx = 0; cx < ncomp; cx++) {
            DST.at(cx) = SRC.at(cx, 0);
          }
        },
        Access::read(this->cdc_project), Access::write(sym))
        ->execute();
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
