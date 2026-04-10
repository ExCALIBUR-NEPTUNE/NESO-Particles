#ifndef _NESO_PARTICLES_ALGORITHMS_SUBDIVIDE_CARTESIAN_CELLS_HPP_
#define _NESO_PARTICLES_ALGORITHMS_SUBDIVIDE_CARTESIAN_CELLS_HPP_

#include "../cartesian_mesh/cartesian_h_mesh.hpp"
#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../particle_spec.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

#include <vector>

namespace NESO::Particles {

/**
 * Implementation to bin particles into subdivision of CartesianHMesh cells. For
 * each mesh cell D subdivisions are performed in each dimension. The subdivided
 * cells are linearly indexed lexicographically from fastest to slowest.
 */
class SubdivideCartesianCells {
protected:
  std::shared_ptr<BufferDevice<int>> d_sub_cell_count;
  std::shared_ptr<BufferDevice<REAL>> d_sub_cell_inverse_widths;
  std::shared_ptr<BufferDevice<REAL>> d_origins;

  template <int ndim, typename GROUP_TYPE>
  static inline void subdivide_cartesian_cells_map(
      std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<INT> sym_name,
      const int sym_component, int const *const RESTRICT k_sub_cell_count,
      REAL const *const RESTRICT k_sub_cell_inverse_widths,
      REAL const *const RESTRICT k_origins) {

    auto particle_group = get_particle_group(particle_sub_group);

    particle_loop(
        "SubdivideCartesianCells::map", particle_sub_group,
        [=](auto INDEX, auto POS, auto SUB_CELL) {
          const auto cell = INDEX.cell;
          const REAL inverse_width = k_sub_cell_inverse_widths[cell];
          const int num_sub_cells = k_sub_cell_count[cell];

          REAL p[ndim];
          for (int dx = 0; dx < ndim; dx++) {
            p[dx] = POS.at(dx);
          }

          int c[ndim];
          for (int dx = 0; dx < ndim; dx++) {
            const REAL origin = k_origins[cell * ndim + dx];
            const REAL offset_P = p[dx] - origin;
            const int unclamped_c = offset_P * inverse_width;
            c[dx] = Kernel::clamp(unclamped_c, 0, num_sub_cells - 1);
          }

          int linear_c = c[ndim - 1];
          for (int dx = (ndim - 2); dx >= 0; dx--) {
            linear_c *= num_sub_cells;
            linear_c += c[dx];
          }

          SUB_CELL.at(sym_component) = linear_c;
        },
        Access::read(ParticleLoopIndex{}),
        Access::read(particle_group->position_dat), Access::write(sym_name))
        ->execute();
  }

public:
  /// Compute device.
  SYCLTargetSharedPtr sycl_target;

  /// The mesh on which the subdivisions are defined.
  CartesianHMeshSharedPtr mesh;

  /// The number of subdivisions per dimension in each cell plus 1.
  std::vector<int> sub_cell_count;

  /**
   * Create new instance with provided mesh and number of subdivisions.
   *
   * @param sycl_target Compute device.
   * @param mesh CartesianHMesh instance to subdivide cells of.
   * @param sub_cell_count Vector containing the number of subdivisions for
   * each mesh cell.
   */
  SubdivideCartesianCells(SYCLTargetSharedPtr sycl_target,
                          CartesianHMeshSharedPtr mesh,
                          std::vector<int> &sub_cell_count);

  /**
   * Map particles to subdivided cells.
   *
   * @param particle_sub_group Particle{Sub}Group of particles.
   * @param sym_name Output Sym in which to store subdivision cell.
   * @param sym_component Output component in which to store subdivision cell.
   */
  template <typename GROUP_TYPE>
  void map(std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<INT> sym_name,
           const int sym_component) {

    auto particle_group = get_particle_group(particle_sub_group);

    NESOASSERT(particle_group->sycl_target.get() == this->sycl_target.get(),
               "SYCLTarget missmatch,");
    NESOASSERT(particle_group->domain->mesh.get() == this->mesh.get(),
               "Domain miss-match between instance and ParticleGroup.");
    NESOASSERT(particle_group->contains_dat(sym_name),
               "Output sym not in ParticleGroup");
    NESOASSERT(sym_component > -1, "Bad output sym component.");
    NESOASSERT(particle_group->get_dat(sym_name)->ncomp > sym_component,
               "Output sym does not have enough components.");

    const int ndim = particle_group->domain->mesh->get_ndim();
    NESOASSERT((0 < ndim) && (ndim < 4), "Unknown number of dimensions.");

    int const *const RESTRICT k_sub_cell_count = this->d_sub_cell_count->ptr;
    REAL const *const RESTRICT k_sub_cell_inverse_widths =
        this->d_sub_cell_inverse_widths->ptr;
    REAL const *const RESTRICT k_origins = this->d_origins->ptr;

    if (ndim == 1) {
      subdivide_cartesian_cells_map<1>(particle_sub_group, sym_name,
                                       sym_component, k_sub_cell_count,
                                       k_sub_cell_inverse_widths, k_origins);
    } else if (ndim == 2) {
      subdivide_cartesian_cells_map<2>(particle_sub_group, sym_name,
                                       sym_component, k_sub_cell_count,
                                       k_sub_cell_inverse_widths, k_origins);
    } else {
      subdivide_cartesian_cells_map<3>(particle_sub_group, sym_name,
                                       sym_component, k_sub_cell_count,
                                       k_sub_cell_inverse_widths, k_origins);
    }
  }
};

extern template void
SubdivideCartesianCells::map(std::shared_ptr<ParticleGroup> particle_sub_group,
                             Sym<INT> sym_name, const int sym_component);

extern template void SubdivideCartesianCells::map(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);

} // namespace NESO::Particles

#endif
