#ifndef _NESO_PARTICLES_EXTERNAL_PETSC_VORONOI_CELL_VOLUME_HPP_
#define _NESO_PARTICLES_EXTERNAL_PETSC_VORONOI_CELL_VOLUME_HPP_

#include "../../algorithms/subdivide_cells_voronoi.hpp"
#include "../../containers/nd_local_array.hpp"
#include "../../device_functions.hpp"
#include "dmplex_interface.hpp"
#include "dmplex_local_mapper.hpp"
#include "dmplex_utility.hpp"

namespace NESO::Particles::PetscInterface {

/**
 * Estimate the volume of Voronoi cells that sub-divide DMPlex cells using Monte
 * Carlo.
 *
 * @param[in] mesh DMPlex mesh interface that describes the base mesh.
 * @param[in] voronoi_cells Voronoi cell instance that describes the Voronoi
 * cells in each mesh cell.
 * @param[in, out] num_samples On call defines the minimum number of samples
 * that must be made per Voronoi cell in each mesh cell. On return contains the
 * number of samples made for each mesh cell.
 * @param[in, out] stol On call sets the tolerance between blocks of samples to
 * use as an exit condition after the number of samples has exceeded the minimum
 * number of samples. On return contains the achieved maximum seen difference
 * between the samples [0,...., i-1] and samples [0,....,i] where i is the
 * number of blocks of samples.
 * @param[in] max_num_samples Maximum number of samples per Voronoi cell to
 * perform.
 * @param[in, out] volumes Output NDLocalArray of Voronoi cell volumes. Will be
 * allocated if too small or nullptr.
 * @param[in, out] rng_in Optionally pass an RNG instance to use for samples.
 * @param[in] default_block_size Optionally specify the desired block size per
 * Voronoi cell for samples.
 */
void estimate_voronoi_cell_volume(DMPlexInterfaceSharedPtr mesh,
                                  SubdivideCellsVoronoiSharedPtr voronoi_cells,
                                  std::size_t &num_samples, REAL &stol,
                                  std::size_t max_num_samples,
                                  NDLocalArraySharedPtr<REAL, 2> &volumes,
                                  std::mt19937 *rng_in = nullptr,
                                  const int default_block_size = 256);

} // namespace NESO::Particles::PetscInterface

#endif
