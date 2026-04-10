#include <neso_particles/algorithms/subdivide_cells_voronoi.hpp>

namespace NESO::Particles {

SubdivideCellsVoronoi::SubdivideCellsVoronoi(SYCLTargetSharedPtr sycl_target,
                                             CellDatSharedPtr<REAL> points)
    : sycl_target(sycl_target), points(points) {

  const int num_cells = points->ncells;
  std::vector<int> h_num_points(num_cells);
  for (int ix = 0; ix < num_cells; ix++) {
    h_num_points.at(ix) = static_cast<int>(points->nrow[ix]);
  }
  this->d_num_points =
      std::make_shared<BufferDevice<int>>(this->sycl_target, h_num_points);
}

template void
SubdivideCellsVoronoi::map(std::shared_ptr<ParticleGroup> particle_sub_group,
                           Sym<INT> sym_name, const int sym_component);

template void
SubdivideCellsVoronoi::map(std::shared_ptr<ParticleSubGroup> particle_sub_group,
                           Sym<INT> sym_name, const int sym_component);
} // namespace NESO::Particles
