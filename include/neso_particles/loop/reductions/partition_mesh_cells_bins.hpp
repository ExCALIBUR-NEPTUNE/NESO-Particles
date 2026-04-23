#ifndef _NESO_PARTICLES_LOOP_REDUCTIONS_PARTITION_MESH_CELLS_BINS_HPP_
#define _NESO_PARTICLES_LOOP_REDUCTIONS_PARTITION_MESH_CELLS_BINS_HPP_

#include "../../containers/index_map.hpp"
#include "../../particle_sub_group/particle_sub_group_base.hpp"

namespace NESO::Particles {

/**
 * Construct a IndexMap where the first key index is the mesh cell that contains
 * the particle and the second key index is provided by an integer property on
 * the particle.
 *
 * @param[in] particle_group ParticleGroup containing particles to partition.
 * @param[in] num_bins Specify the number of possible bins.
 * @param[in] bin_sym Integer Sym that contains the bin the particle is in for
 * the second key of the index map.
 * @param[in] bin_component Specification of which component of the bin_sym that
 * should be used for binning the particle.
 * @param[in, out] index_map IndexMap to populate with particle layers.
 */
void partition_mesh_cells_bins(ParticleGroupSharedPtr particle_group,
                               const int num_bins, Sym<INT> bin_sym,
                               const int bin_component,
                               IndexMapSharedPtr<2, 1> index_map);

/**
 * Construct a IndexMap where the first key index is the mesh cell that contains
 * the particle and the second key index is provided by an integer property on
 * the particle.
 *
 * @param[in] particle_sub_group ParticleGroup containing particles to
 * partition.
 * @param[in] num_bins Specify the number of possible bins.
 * @param[in] bin_sym Integer Sym that contains the bin the particle is in for
 * the second key of the index map.
 * @param[in] bin_component Specification of which component of the bin_sym that
 * should be used for binning the particle.
 * @param[in, out] index_map IndexMap to populate with particle layers.
 */
void partition_mesh_cells_bins(ParticleSubGroupSharedPtr particle_sub_group,
                               const int num_bins, Sym<INT> bin_sym,
                               const int bin_component,
                               IndexMapSharedPtr<2, 1> index_map);

} // namespace NESO::Particles

#endif
