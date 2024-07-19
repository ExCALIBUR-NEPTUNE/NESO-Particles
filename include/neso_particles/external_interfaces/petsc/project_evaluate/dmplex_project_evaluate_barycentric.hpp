#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BARYCENTRIC_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BARYCENTRIC_H_

#include "../../common/coordinate_mapping.hpp"
#include "dmplex_project_evaluate_base.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * TODO
 */
class DMPlexProjectEvaluateBarycentric : public DMPlexProjectEvaluateBase {
protected:
  std::shared_ptr<CellDatConst<REAL>> cdc_project;
  std::shared_ptr<CellDatConst<REAL>> cdc_volumes;
  std::shared_ptr<CellDatConst<int>> cdc_num_vertices;
  std::shared_ptr<CellDatConst<REAL>> cdc_vertices;
  std::shared_ptr<CellDatConst<REAL>> cdc_matrices;
  DMPlexInterfaceSharedPtr mesh;

  inline void check_setup() {
    NESOASSERT(this->qpm->points_added(),
               "QuadraturePointMapper needs points adding to it.");
  }
  inline void check_ncomp(const int ncomp) {
    if (this->cdc_project->nrow < ncomp) {
      const int cell_count = this->qpm->domain->mesh->get_cell_count();
      this->cdc_project = std::make_shared<CellDatConst<REAL>>(
          this->qpm->sycl_target, cell_count, ncomp, 3);
    }
  }

  inline void setup_matrices() {
    if (this->cdc_matrices == nullptr) {
      NESOASSERT(this->qpm->points_added(),
                 "Requires the QuadraturePointMapper to be fully setup.");
      const int cell_count = this->qpm->domain->mesh->get_cell_count();
      this->cdc_matrices = std::make_shared<CellDatConst<REAL>>(
          this->qpm->sycl_target, cell_count, 3, 3);
      this->cdc_matrices->fill(-1000.0);
      auto position_dat = this->qpm->particle_group->position_dat;

      particle_loop(
          this->qpm->particle_group,
          [=](auto VERTICES, auto POS, auto MASK, auto MATRIX) {
            const auto quad_index = MASK.at(3);
            if ((-1 < quad_index) && (quad_index < 3)) {
              REAL l1, l2, l3;
              const REAL x = POS.at(0);
              const REAL y = POS.at(1);
              ExternalCommon::triangle_cartesian_to_barycentric(
                  VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
                  VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1), x, y,
                  &l1, &l2, &l3);
              MATRIX.at(quad_index, 0) = l1;
              MATRIX.at(quad_index, 1) = l2;
              MATRIX.at(quad_index, 2) = l3;
            }
          },
          Access::read(this->cdc_vertices), Access::read(position_dat),
          Access::read(Sym<INT>("ADDING_RANK_INDEX")),
          Access::write(this->cdc_matrices))
          ->execute();

      // compute the inverses
      REAL L[9], M[9];
      for (int cellx = 0; cellx < cell_count; cellx++) {
        for (int rx = 0; rx < 3; rx++) {
          for (int cx = 0; cx < 3; cx++) {
            REAL v = this->cdc_matrices->get_value(cellx, rx, cx);
            if (v < 0) {
              v = 0.0;
            };
            if (v > 1.0) {
              v = 1.0;
            };
            NESOASSERT(((0.0 <= v) && (v <= 1.0)),
                       "Bad Barycentric coordinate for quadrature point.");
            L[rx * 3 + cx] = v;
          }
        }
        ExternalCommon::triangle_barycentric_invert(L, M);
        for (int rx = 0; rx < 3; rx++) {
          for (int cx = 0; cx < 3; cx++) {
            this->cdc_matrices->set_value(cellx, rx, cx, M[rx * 3 + cx]);
          }
        }
      }
    }
  }

public:
  ExternalCommon::QuadraturePointMapperSharedPtr qpm;
  std::string function_space;
  int polynomial_order;
  DMPlexProjectEvaluateBarycentric() = default;

  /**
   * TODO
   */
  virtual inline std::vector<VTK::UnstructuredCell> get_vtk_data() override {
    const int cell_count = this->mesh->get_cell_count();
    std::vector<VTK::UnstructuredCell> data(cell_count);
    NESOASSERT(false, "not implemented");
    return data;
  }

  /**
   * TODO
   */
  DMPlexProjectEvaluateBarycentric(
      ExternalCommon::QuadraturePointMapperSharedPtr qpm,
      std::string function_space, int polynomial_order,
      const bool testing = false)
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
        this->qpm->sycl_target, cell_count, 1, 4);
    this->cdc_num_vertices = std::make_shared<CellDatConst<int>>(
        this->qpm->sycl_target, cell_count, 1, 1);
    this->cdc_vertices = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 4, 2);
    this->cdc_matrices = nullptr;

    // For each cell record the verticies.
    std::vector<std::vector<REAL>> vertices;
    for (int cx = 0; cx < cell_count; cx++) {
      this->mesh->dmh->get_cell_vertices(cx, vertices);
      const int num_verts = vertices.size();
      // TODO implement for quads
      NESOASSERT(testing || ((2 < num_verts) && (num_verts < 5)),
                 "Unexpected number of vertices (expected 3 or 4).");
      this->cdc_num_vertices->set_value(cx, 0, 0, num_verts);
      for (int vx = 0; vx < num_verts; vx++) {
        for (int dx = 0; dx < 2; dx++) {
          this->cdc_vertices->set_value(cx, vx, dx, vertices.at(vx).at(dx));
        }
      }
    }
    this->cdc_volumes = std::make_shared<CellDatConst<REAL>>(
        this->qpm->sycl_target, cell_count, 1, 1);

    // For each cell record the volume.
    for (int cx = 0; cx < cell_count; cx++) {
      const auto volume = this->mesh->dmh->get_cell_volume(cx);
      this->cdc_volumes->set_value(
          cx, 0, 0, this->cdc_num_vertices->get_value(cx, 0, 0) / volume);
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
    this->check_ncomp(ncomp);

    auto destination_dat = this->qpm->get_sym(ncomp);
    auto destination_position_dat = this->qpm->particle_group->position_dat;

    // DG0 projection onto the CellDatConst
    this->cdc_project->fill(0.0);
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::project_0", particle_group,
        [=](auto POS, auto VERTICES, auto SRC, auto DST) {
          REAL l1, l2, l3;
          const REAL x = POS.at(0);
          const REAL y = POS.at(1);
          ExternalCommon::triangle_cartesian_to_barycentric(
              VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
              VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1), x, y,
              &l1, &l2, &l3);
          const REAL value = SRC.at(0);
          for (int cx = 0; cx < ncomp; cx++) {
            DST.fetch_add(cx, 0, l1 * SRC.at(cx));
            DST.fetch_add(cx, 1, l2 * SRC.at(cx));
            DST.fetch_add(cx, 2, l3 * SRC.at(cx));
          }
        },
        Access::read(position_dat), Access::read(this->cdc_vertices),
        Access::read(sym), Access::add(this->cdc_project))
        ->execute();

    // Read the CellDatConst values onto the quadrature point values
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::project_1",
        this->qpm->particle_group,
        [=](auto POS, auto VERTICES, auto SRC, auto VOLUMES, auto DST) {
          REAL l1, l2, l3;
          const REAL x = POS.at(0);
          const REAL y = POS.at(1);
          ExternalCommon::triangle_cartesian_to_barycentric(
              VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
              VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1), x, y,
              &l1, &l2, &l3);
          const REAL iv = VOLUMES.at(0, 0);
          for (int cx = 0; cx < ncomp; cx++) {
            DST.at(cx) = iv * (SRC.at(cx, 0) * l1 + SRC.at(cx, 1) * l2 +
                               SRC.at(cx, 2) * l3);
          }
        },
        Access::read(destination_position_dat),
        Access::read(this->cdc_vertices), Access::read(this->cdc_project),
        Access::read(this->cdc_volumes), Access::write(destination_dat))
        ->execute();
  }

  /**
   * TODO
   * Evaluates the function defined by the values in the QuadraturePointValues
   * at the particle locations.
   */
  inline void evaluate(ParticleGroupSharedPtr particle_group,
                       Sym<REAL> sym) override {

    /*
      If:
        1) Quadrature point i has Barycentric coordinates li1, li2, li3 and
      evaluation ui. 2) The vertices have values q1, q2 and q3.

      Then
        | l11 l12 l13 |   |q1|   |u1|
        | l21 l22 l23 | x |q2| = |u2|
        | l31 l32 l33 |   |q3|   |u3|
    */
    this->setup_matrices();

    auto dat = particle_group->get_dat(sym);
    auto position_dat = particle_group->position_dat;

    const int ncomp = dat->ncomp;
    this->check_ncomp(ncomp);
    auto source_dat = this->qpm->get_sym(ncomp);
    this->cdc_project->fill(0.0);

    // Copy from the quadrature point values into the CellDatConst
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::evaluate_0",
        this->qpm->particle_group,
        [=](auto MATRIX, auto SRC, auto DST, auto MASK) {
          /*
            Using the identity

            | m11 m12 m13 |   |u_1|   | m11 m12 m13 |   |u_1|
            | m21 m22 m23 | x |u_2| = | m21 m22 m23 | x | 0 | +
            | m31 m32 m33 |   |u_3|   | m31 m32 m33 |   | 0 |

                                      | m11 m12 m13 |   | 0 |
                                      | m21 m22 m23 | x |u_2| +
                                      | m31 m32 m33 |   | 0 |

                                      | m11 m12 m13 |   | 0 |
                                      | m21 m22 m23 | x | 0 |
                                      | m31 m32 m33 |   |u_3|

            quad_index is a row in the above matrix equation. Hence this kernel
            is performing

            | m11 m12 m13 |   | 0 |   |v1|
            | m21 m22 m23 | x |u_i| = |v2|
            | m31 m32 m33 |   | 0 |   |v3|

            where i is quad_index. Hence the kernel scales the i^th column by
            the contribution. Then atomically adding the v1,v2,v3 values across
            all the quad indices.

           */
          const auto quad_index = MASK.at(3);

          if ((-1 < quad_index) && (quad_index < 3)) {
            for (int cx = 0; cx < ncomp; cx++) {
              const REAL value = SRC.at(cx);

              const REAL v1 = MATRIX.at(0, quad_index) * value;
              const REAL v2 = MATRIX.at(1, quad_index) * value;
              const REAL v3 = MATRIX.at(2, quad_index) * value;
              DST.fetch_add(cx, 0, v1);
              DST.fetch_add(cx, 1, v2);
              DST.fetch_add(cx, 2, v3);
            }
          }
        },
        Access::read(this->cdc_matrices), Access::read(source_dat),
        Access::add(this->cdc_project),
        Access::read(Sym<INT>("ADDING_RANK_INDEX")))
        ->execute();

    // Copy from the CellDatConst to the output particle group
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::evaluate_1", particle_group,
        [=](auto POS, auto VERTICES, auto SRC, auto DST) {
          REAL l1, l2, l3;
          const REAL x = POS.at(0);
          const REAL y = POS.at(1);
          ExternalCommon::triangle_cartesian_to_barycentric(
              VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
              VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1), x, y,
              &l1, &l2, &l3);

          for (int cx = 0; cx < ncomp; cx++) {
            DST.at(cx) =
                SRC.at(cx, 0) * l1 + SRC.at(cx, 1) * l2 + SRC.at(cx, 2) * l3;
          }
        },
        Access::read(position_dat), Access::read(this->cdc_vertices),
        Access::read(this->cdc_project), Access::write(sym))
        ->execute();
  }
};

} // namespace NESO::Particles::PetscInterface

#endif
