#ifndef _NESO_PARTICLES_PAIR_LOOP_PAIR_LOOP_HPP_
#define _NESO_PARTICLES_PAIR_LOOP_PAIR_LOOP_HPP_

/**
 * @defgroup particle_pair_loop Particle Pair Loop
 * @details This section contains documentation for particle pair looping.
 * Particle pair looping is a looping type similar to particle loop except that
 * the kernel operates on two particles.
 * @np_rst_source_file{particle_pair_loop_group_description.rst}
 */

/**
 * @defgroup particle_pair_loop_functions Particle Pair Loop Functions
 * @ingroup particle_pair_loop
 * @details Functions that create Particle Pair Loops. These functions accept an
 * absolute pair list as the iteration set specification.
 */

#include "cellwise_pair_list.hpp"
#include "cellwise_pair_list_absolute.hpp"
#include "cellwise_pair_list_block.hpp"
#include "cellwise_pair_list_host.hpp"
#include "cellwise_pair_list_simple.hpp"
#include "pair_utility.hpp"
#include "particle_pair_loop_args.hpp"
#include "particle_pair_loop_base.hpp"
#include "particle_pair_loop_cellwise_pair_list.hpp"
#include "particle_pair_loop_cellwise_pair_list_block.hpp"

#endif
