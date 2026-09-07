#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_VORONOI_CELL_VOLUME_HPP_
#define _NESO_PARTICLES_EXTERNAL_PETSC_VORONOI_CELL_VOLUME_HPP_

#include "../../algorithms/subdivide_cells_voronoi.hpp"
#include "../../device_functions.hpp"
#include "dmplex_interface.hpp"
#include "dmplex_local_mapper.hpp"
#include "dmplex_utility.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * TODO
 */
void estimate_voronoi_cell_volume(DMPlexInterfaceSharedPtr mesh,
                                  SubdivideCellsVoronoiSharedPtr voronoi_cells,
                                  std::size_t &num_samples, REAL &stol,
                                  std::size_t &max_num_samples,
                                  CellDatSharedPtr<REAL> volumes,
                                  std::mt19937 *rng_in = nullptr,
                                  const int default_block_size = 4096);

} // namespace NESO::Particles::PetscInterface

#endif
