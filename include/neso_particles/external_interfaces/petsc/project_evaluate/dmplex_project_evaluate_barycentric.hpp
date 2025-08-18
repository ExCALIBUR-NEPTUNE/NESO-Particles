#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BARYCENTRIC_H_
#define _NESO_PARTICLES_EXTERNAL_PETSC_DMPLEX_PROJECT_EVALUATE_BARYCENTRIC_H_

#include "../../common/coordinate_mapping.hpp"
#include "../petsc_utility.hpp"
#include "dmplex_project_evaluate_base.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Implementation to deposit particle data into and evalute from DG1 function
 * spaces by using Barycentric interpolation.
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
          this->qpm->sycl_target, cell_count, ncomp, 4);
    }
  }

  inline void setup_matrices() {
    if (this->cdc_matrices == nullptr) {
      NESOASSERT(this->qpm->points_added(),
                 "Requires the QuadraturePointMapper to be fully setup.");
      // Validate the QPM
      const int cell_count = this->qpm->domain->mesh->get_cell_count();
      auto cdc_check_qpm = std::make_shared<CellDatConst<int>>(
          this->qpm->sycl_target, cell_count, 5, 1);
      cdc_check_qpm->fill(0);

      particle_loop(
          this->qpm->particle_group,
          [&](auto CHECK_QPM, auto NUM_VERTICES, auto MASK) {
            const auto quad_index = MASK.at(3);
            const auto num_vertices = NUM_VERTICES.at(0, 0);
            if ((-1 < quad_index) && (quad_index < num_vertices)) {
              CHECK_QPM.fetch_add(0, 0, 1);
              CHECK_QPM.fetch_add(quad_index + 1, 0, 1);
            }
          },
          Access::add(cdc_check_qpm), Access::read(this->cdc_num_vertices),
          Access::read(Sym<INT>("ADDING_RANK_INDEX")))
          ->execute();

      for (int cellx = 0; cellx < cell_count; cellx++) {
        auto c = cdc_check_qpm->get_cell(cellx);
        auto num_vertices = this->cdc_num_vertices->get_value(cellx, 0, 0);
        NESOASSERT(c->at(0, 0) == num_vertices,
                   "Number of quadrature points found is not the required "
                   "number for this element. Found: " +
                       std::to_string(c->at(0, 0)) + " expected " +
                       std::to_string(num_vertices) + ".");
        for (int vx = 0; vx < num_vertices; vx++) {
          NESOASSERT(c->at(vx + 1, 0) == 1,
                     "Duplicate quadrature point indices detected.");
        }
      }

      this->compute_barycentric_coordinates(this->qpm->particle_group);

      this->cdc_matrices = std::make_shared<CellDatConst<REAL>>(
          this->qpm->sycl_target, cell_count, 4, 4);
      // These bary centric values sum to 1 and are positive hence this is a
      // valid mask value.
      const REAL mask_value = -100.0;
      this->cdc_matrices->fill(mask_value);

      particle_loop(
          this->qpm->particle_group,
          [=](auto NUM_VERTICES, auto B, auto MASK, auto MATRIX) {
            const auto quad_index = MASK.at(3);
            const auto num_vertices = NUM_VERTICES.at(0, 0);
            if ((-1 < quad_index) && (quad_index < num_vertices)) {
              for (int vx = 0; vx < num_vertices; vx++) {
                const auto lx = B.at(vx);
                MATRIX.at(quad_index, vx) = lx;
              }
            }
          },
          Access::read(this->cdc_num_vertices),
          Access::read(sym_barycentric_coords),
          Access::read(Sym<INT>("ADDING_RANK_INDEX")),
          Access::write(this->cdc_matrices))
          ->execute();

      // compute the inverses
      REAL L[16], M[16];
      for (int cellx = 0; cellx < cell_count; cellx++) {
        const auto num_vertices =
            this->cdc_num_vertices->get_value(cellx, 0, 0);
        const auto Mdata = this->cdc_matrices->get_cell(cellx);
        for (int rx = 0; rx < num_vertices; rx++) {
          for (int cx = 0; cx < num_vertices; cx++) {
            REAL v = Mdata->at(rx, cx);
            NESOASSERT(v != mask_value,
                       "Bad Barycentric coordinate for quadrature point.");
            v = std::max(0.0, std::min(1.0, v));
            L[rx * num_vertices + cx] = v;
          }
        }
        PetscInterface::invert_matrix(num_vertices, L, M);
        for (int rx = 0; rx < num_vertices; rx++) {
          for (int cx = 0; cx < num_vertices; cx++) {
            Mdata->at(rx, cx) = M[rx * num_vertices + cx];
          }
        }
        this->cdc_matrices->set_cell(cellx, Mdata);
      }
    }
  }

  template <typename T>
  inline void
  compute_barycentric_coordinates(std::shared_ptr<T> particle_sub_group) {
    auto particle_group = get_particle_group(particle_sub_group);

    // Add the particle property for the Barycentric coordinates if it does not
    // exist already.
    if (!particle_group->contains_dat(sym_barycentric_coords)) {
      particle_group->add_particle_dat(sym_barycentric_coords,
                                       ncomp_barycentric_coords);
    }
    // Compute the Barycentric coordinates for all the particles in the
    // ParticleGroup.
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::compute_barycentric_coordinates",
        particle_sub_group,
        [=](auto P, auto B, auto NUM_VERTICES, auto VERTICES) {
          const int num_vertices = NUM_VERTICES.at(0, 0);
          if (num_vertices == 3) {
            // Is triangle
            ExternalCommon::triangle_cartesian_to_barycentric(
                VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
                VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1),
                P.at(0), P.at(1), &B.at(0), &B.at(1), &B.at(2));
          } else {
            // Is quad
            ExternalCommon::quad_cartesian_to_barycentric(
                VERTICES.at(0, 0), VERTICES.at(0, 1), VERTICES.at(1, 0),
                VERTICES.at(1, 1), VERTICES.at(2, 0), VERTICES.at(2, 1),
                VERTICES.at(3, 0), VERTICES.at(3, 1), P.at(0), P.at(1),
                &B.at(0), &B.at(1), &B.at(2), &B.at(3));
          }
        },
        Access::read(particle_group->position_dat),
        Access::write(sym_barycentric_coords),
        Access::read(this->cdc_num_vertices), Access::read(this->cdc_vertices))
        ->execute();
  }

  template <typename T>
  inline void evaluate_inner(std::shared_ptr<T> particle_sub_group,
                             Sym<REAL> sym) {

    /*
      If:
        1) Quadrature point i has Barycentric coordinates li1, li2, li3 and
      evaluation ui. 2) The vertices have values q1, q2 and q3.

      Then
        | l11 l12 l13 |   |q1|   |u1|
        | l21 l22 l23 | x |q2| = |u2|
        | l31 l32 l33 |   |q3|   |u3|
    */
    this->compute_barycentric_coordinates(particle_sub_group);
    auto dat = get_particle_group(particle_sub_group)->get_dat(sym);

    const int ncomp = dat->ncomp;
    this->check_ncomp(ncomp);
    auto source_dat = this->qpm->get_sym(ncomp);
    this->cdc_project->fill(0.0);

    // Copy from the quadrature point values into the CellDatConst
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::evaluate_0",
        this->qpm->particle_group,
        [=](auto NUM_VERTICES, auto MATRIX, auto SRC, auto DST, auto MASK) {
          /*
            Using the identity

            | m11 m12 m13 |  |u_1|   | m11 m12 m13 |  |u_1|
            | m21 m22 m23 |  |u_2| = | m21 m22 m23 |  | 0 | +
            | m31 m32 m33 |  |u_3|   | m31 m32 m33 |  | 0 |

                                     | m11 m12 m13 |  | 0 |
                                     | m21 m22 m23 |  |u_2| +
                                     | m31 m32 m33 |  | 0 |

                                     | m11 m12 m13 |  | 0 |
                                     | m21 m22 m23 |  | 0 |
                                     | m31 m32 m33 |  |u_3|

            quad_index is a row in the above matrix equation. Hence this kernel
            is performing

            | m11 m12 m13 |  | 0 |   |u_i m1i|
            | m21 m22 m23 |  |u_i| = |u_i m2i|
            | m31 m32 m33 |  | 0 |   |u_i m3i|

            where i is quad_index. Hence the kernel scales the i^th column by
            the contribution. Then atomically adding the values across all the
            quad indices.
           */
          const auto num_vertices = NUM_VERTICES.at(0, 0);
          const auto quad_index = MASK.at(3);
          if ((-1 < quad_index) && (quad_index < num_vertices)) {
            for (int vx = 0; vx < num_vertices; vx++) {
              for (int cx = 0; cx < ncomp; cx++) {
                const REAL value = SRC.at(cx);
                const REAL weight = MATRIX.at(vx, quad_index) * value;
                DST.combine(cx, vx, weight);
              }
            }
          }
        },
        Access::read(this->cdc_num_vertices), Access::read(this->cdc_matrices),
        Access::read(source_dat),
        Access::reduce(this->cdc_project, Kernel::plus<REAL>()),
        Access::read(Sym<INT>("ADDING_RANK_INDEX")))
        ->execute();

    // Copy from the CellDatConst to the output particle group
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::evaluate_1", particle_sub_group,
        [=](auto B, auto NUM_VERTICES, auto SRC, auto DST) {
          const auto num_vertices = NUM_VERTICES.at(0, 0);
          for (int cx = 0; cx < ncomp; cx++) {
            REAL tmp = 0.0;
            for (int vx = 0; vx < num_vertices; vx++) {
              tmp += B.at(vx) * SRC.at(cx, vx);
            }
            DST.at(cx) = tmp;
          }
        },
        Access::read(sym_barycentric_coords),
        Access::read(this->cdc_num_vertices), Access::read(this->cdc_project),
        Access::write(sym))
        ->execute();
  }

  template <typename T>
  inline void project_inner(std::shared_ptr<T> particle_sub_group,
                            Sym<REAL> sym) {
    auto dat = get_particle_group(particle_sub_group)->get_dat(sym);

    const int ncomp = dat->ncomp;
    this->check_ncomp(ncomp);

    auto destination_dat = this->qpm->get_sym(ncomp);
    this->compute_barycentric_coordinates(particle_sub_group);

    // Barycentric interpolation onto the CellDatConst
    this->cdc_project->fill(0.0);
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::project_0", particle_sub_group,
        [=](auto B, auto NUM_VERTICES, auto SRC, auto DST) {
          const int num_vertices = NUM_VERTICES.at(0, 0);
          for (int vx = 0; vx < num_vertices; vx++) {
            for (int cx = 0; cx < ncomp; cx++) {
              const REAL lx = B.at(vx);
              DST.combine(cx, vx, lx * SRC.at(cx));
            }
          }
        },
        Access::read(sym_barycentric_coords),
        Access::read(this->cdc_num_vertices), Access::read(sym),
        Access::reduce(this->cdc_project, Kernel::plus<REAL>()))
        ->execute();

    // Read the CellDatConst values onto the quadrature point values
    particle_loop(
        "DMPlexProjectEvaluateBarycentric::project_1",
        this->qpm->particle_group,
        [=](auto B, auto NUM_VERTICES, auto SRC, auto VOLUMES, auto DST,
            auto MASK) {
          const int num_vertices = NUM_VERTICES.at(0, 0);
          const auto quad_index = MASK.at(3);
          if ((-1 < quad_index) && (quad_index < num_vertices)) {
            const REAL iv = VOLUMES.at(0, 0);
            for (int cx = 0; cx < ncomp; cx++) {
              REAL tmp_accumulation = 0.0;
              for (int vx = 0; vx < num_vertices; vx++) {
                tmp_accumulation += B.at(vx) * SRC.at(cx, vx);
              }
              DST.at(cx) = iv * tmp_accumulation;
            }
          }
        },
        Access::read(sym_barycentric_coords),
        Access::read(this->cdc_num_vertices), Access::read(this->cdc_project),
        Access::read(this->cdc_volumes), Access::write(destination_dat),
        Access::read(Sym<INT>("ADDING_RANK_INDEX")))
        ->execute();
  }

public:
  ExternalCommon::QuadraturePointMapperSharedPtr qpm;
  std::string function_space;
  int polynomial_order;
  virtual ~DMPlexProjectEvaluateBarycentric() = default;
  DMPlexProjectEvaluateBarycentric() = default;
  static inline const Sym<REAL> sym_barycentric_coords =
      Sym<REAL>("NESO_PARTICLES_DMPLEX_PROJ_EVAL_BARY_COORDS");
  static inline constexpr int ncomp_barycentric_coords = 4;

  /**
   * Get a representation of the internal state which can be passed to the
   * VTKHDF writer.
   *
   * @returns Data for VTKHDF unstructured grid writer.
   */
  virtual std::vector<VTK::UnstructuredCell> get_vtk_data() override;

  /**
   * Create a DG1 project/evaluate instance from a QuadraturePointMapper which
   * uses Barycentric interpolation.
   *
   * @param qpm QuadraturePointMapper with a single point per cell.
   * @param function_space Should be "Barycentric".
   * @param polynomial_order Should be 1.
   */
  DMPlexProjectEvaluateBarycentric(
      ExternalCommon::QuadraturePointMapperSharedPtr qpm,
      std::string function_space, int polynomial_order,
      const bool testing = false);

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
};

} // namespace NESO::Particles::PetscInterface

#endif
