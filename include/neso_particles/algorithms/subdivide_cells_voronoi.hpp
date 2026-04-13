#ifndef _NESO_PARTICLES_ALGORITHMS_SUBDIVIDE_CELLS_VORONOI_HPP_
#define _NESO_PARTICLES_ALGORITHMS_SUBDIVIDE_CELLS_VORONOI_HPP_

#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../particle_spec.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

/**
 * Implementation for subdividing mesh cells into voronoi cells defined by
 * points.
 */
class SubdivideCellsVoronoi {
protected:
  // Number of points in each cell.
  std::shared_ptr<BufferDevice<int>> d_num_points;
  std::vector<int> h_num_points;

  template <int ndim, typename GROUP_TYPE>
  static inline void
  subdivide_cells_voronoi_map(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                              Sym<INT> sym_name, const int sym_component,
                              int const *const RESTRICT k_num_points,
                              CellDatSharedPtr<REAL> points) {

    auto particle_group = get_particle_group(particle_sub_group);

    particle_loop(
        "SubdivideCellsVoronoi::map", particle_sub_group,
        [=](auto INDEX, auto POS, auto VCELL, auto POINTS) {
          REAL p[ndim];
          for (int dx = 0; dx < ndim; dx++) {
            p[dx] = POS.at(dx);
          }

          const auto cell = INDEX.cell;
          const int num_points_in_cell = k_num_points[cell];

          int vcell = 0;
          constexpr REAL default_min_distance =
              std::numeric_limits<REAL>::max();
          REAL min_distance = default_min_distance;

          for (int pointx = 0; pointx < num_points_in_cell; pointx++) {
            REAL distance_squared = 0.0;
            for (int dx = 0; dx < ndim; dx++) {
              const REAL r = p[dx] - POINTS.at(pointx, dx);
              distance_squared = Kernel::fma(r, r, distance_squared);
            }

            if (distance_squared < min_distance) {
              min_distance = distance_squared;
              vcell = pointx;
            }
          }

          VCELL.at(sym_component) = vcell;
        },
        Access::read(ParticleLoopIndex{}),
        Access::read(particle_group->position_dat), Access::write(sym_name),
        Access::read(points))
        ->execute();
  }

public:
  /// Compute device.
  SYCLTargetSharedPtr sycl_target;
  /// Voronoi cell points.
  CellDatSharedPtr<REAL> points;

  /**
   * Create new instance using a compute device and set of points. For a
   * D-dimenionsal mesh N points should be specified as a NxD matrix per mesh
   * cell. i.e. point per row.
   *
   * @param sycl_target Compute target.
   * @param points Points, given mesh cell-wise, that define the Voronoi cells.
   */
  SubdivideCellsVoronoi(SYCLTargetSharedPtr sycl_target,
                        CellDatSharedPtr<REAL> points);

  /**
   * Map particles to Voronoi cells. If no Voronoi cells are specified for a
   * mesh cell then the particles in that mesh cell are assigned to the 0-th
   * Voronoi cell.
   *
   * @param particle_sub_group Particle{Sub}Group of particles.
   * @param sym_name Output Sym in which to store Voronoi cell.
   * @param sym_component Output component in which to store Voronoi cell.
   */
  template <typename GROUP_TYPE>
  void map(std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<INT> sym_name,
           const int sym_component) {

    auto particle_group = get_particle_group(particle_sub_group);

    NESOASSERT(particle_group->sycl_target.get() == this->sycl_target.get(),
               "SYCLTarget missmatch,");
    NESOASSERT(particle_group->domain->mesh->get_cell_count() ==
                   this->points->ncells,
               "Miss-match of cell count in points and cell count of particle "
               "domain.");
    NESOASSERT(particle_group->contains_dat(sym_name),
               "Output sym not in ParticleGroup");
    NESOASSERT(sym_component > -1, "Bad output sym component.");
    NESOASSERT(particle_group->get_dat(sym_name)->ncomp > sym_component,
               "Output sym does not have enough components.");

    int const *const RESTRICT k_num_points = this->d_num_points->ptr;

    const int ndim = particle_group->domain->mesh->get_ndim();
    NESOASSERT((0 < ndim) && (ndim < 4), "Unknown number of dimensions.");

    if (ndim == 1) {
      subdivide_cells_voronoi_map<1>(particle_sub_group, sym_name,
                                     sym_component, k_num_points, this->points);
    } else if (ndim == 2) {
      subdivide_cells_voronoi_map<2>(particle_sub_group, sym_name,
                                     sym_component, k_num_points, this->points);
    } else {
      subdivide_cells_voronoi_map<3>(particle_sub_group, sym_name,
                                     sym_component, k_num_points, this->points);
    }
  }

  /**
   * @returns The number of Voronoi cells in each mesh cell.
   */
  const std::vector<int> &get_num_subdivision_cells();
};

extern template void
SubdivideCellsVoronoi::map(std::shared_ptr<ParticleGroup> particle_sub_group,
                           Sym<INT> sym_name, const int sym_component);

extern template void
SubdivideCellsVoronoi::map(std::shared_ptr<ParticleSubGroup> particle_sub_group,
                           Sym<INT> sym_name, const int sym_component);

} // namespace NESO::Particles

#endif
