#ifndef __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_PAIR_LIST_ABSOLUTE_HPP_
#define __NESO_PARTICLES_ALGORITHMS_DSMC_CELLWISE_PAIR_LIST_ABSOLUTE_HPP_

#include "../particle_group.hpp"
#include "cellwise_pair_list.hpp"
#include "cellwise_pair_list_block.hpp"

namespace NESO::Particles {

/**
 * Type for defining the A and B sets along with the pair list.
 */
template <typename GROUP_TYPE, typename PAIR_LIST_TYPE>
struct CellwisePairListAbsolute;
/**
 * Type for defining the A and B sets along with the pair list where both A and
 * B are ParticleGroups.
 */
template <> struct CellwisePairListAbsolute<ParticleGroup, CellwisePairList> {
  ParticleGroupSharedPtr A;
  ParticleGroupSharedPtr B;
  CellwisePairListSharedPtr pair_list;
  CellwisePairListAbsolute<ParticleGroup, CellwisePairList>() = default;
  ~CellwisePairListAbsolute<ParticleGroup, CellwisePairList>() = default;

  /**
   * Create cell wise pair list where the pair indices correspond to particle
   * layers in two ParticleGroups A and B. A may equal B.
   *
   * @param A First ParticleGroup which suppies the "i" particles.
   * @param B Second ParticleGroup which suppies the "j" particles.
   * @param pair_list Pair list of particle pairs.
   */
  CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      ParticleGroupSharedPtr A, ParticleGroupSharedPtr B,
      CellwisePairListSharedPtr pair_list)
      : A(A), B(B), pair_list(pair_list) {}
};

/**
 * Type for defining the A and B sets along with the pair list where both A and
 * B are ParticleGroups.
 */
template <>
struct CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface> {
  ParticleGroupSharedPtr A;
  ParticleGroupSharedPtr B;
  CellwisePairListBlockInterfaceSharedPtr pair_list;
  CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>() =
      default;
  ~CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>() =
      default;

  /**
   * Create cell wise pair list where the pair indices correspond to particle
   * layers in two ParticleGroups A and B. A may equal B.
   *
   * @param A First ParticleGroup which suppies the "i" particles.
   * @param B Second ParticleGroup which suppies the "j" particles.
   * @param pair_list Pair list of particle pairs.
   */
  CellwisePairListAbsolute<ParticleGroup, CellwisePairListBlockInterface>(
      ParticleGroupSharedPtr A, ParticleGroupSharedPtr B,
      CellwisePairListBlockInterfaceSharedPtr pair_list)
      : A(A), B(B), pair_list(pair_list) {}
};

} // namespace NESO::Particles

#endif
