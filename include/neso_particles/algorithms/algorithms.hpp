#ifndef _NESO_PARTICLES_ALGORITHMS_ALGORITHMS_HPP_
#define _NESO_PARTICLES_ALGORITHMS_ALGORITHMS_HPP_

/**
 * @defgroup algorithms Algorithms
 * @{
 * @details This section contains functionality which is particle-adjacent. For
 * example numerical methods used by particle based models and implementations
 * of methods which use particles.
 * @}
 */

/**
 * @defgroup algorithms_general General
 * @ingroup algorithms
 * @details This section contains commonly desired functionality that is
 * implemented already in the library.
 */

/**
 * @defgroup algorithms_dsmc DSMC
 * @ingroup algorithms
 * @details This section contains functionality for implementing DSMC style
 * methods.
 */

#include "cellwise_methods.hpp"
#include "common.hpp"
#include "dsmc/dsmc.hpp"
#include "integration.hpp"
#include "nd_local_array_looping.hpp"
#include "particle_data_movement.hpp"
#include "reduce_dat_cellwise.hpp"
#include "subdivide_cartesian_cells.hpp"
#include "subdivide_cells_voronoi.hpp"
#include "unseen_value_extractor.hpp"

#endif
