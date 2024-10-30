#ifndef _NESO_PARTICLES_PARTICLE_LOOP_INDEX_H_
#define _NESO_PARTICLES_PARTICLE_LOOP_INDEX_H_

#include "../particle_group.hpp"
#include "particle_loop_base.hpp"

namespace NESO::Particles {

/**
 * The type to pass to a ParticleLoop to read the ParticleLoop loop index in a
 * kernel.
 */
struct ParticleLoopIndex {};

/**
 * Defines the access type for the cell, layer indexing.
 */
namespace Access::LoopIndex {
/**
 * ParticleLoop index containing the cell and layer.
 */
struct Read {
  /// The cell containing the particle.
  INT cell;
  /// The layer of the particle.
  INT layer;
  /// The looping layer of the particle.
  INT loop_layer;
  /// The type of the ParticleLoop (intended for internal use).
  int loop_type_int;
  /// pointer to the exclusive sum of particle counts in each cell (intended for
  /// internal use).
  INT const *npart_cell_es;
  /// pointer to the exclusive sum of particle counts in each cell for loop
  /// bounds (intended for internal use).
  INT const *npart_cell_es_lb;
  /// Loop iteration index (intended for internal use - see
  /// get_loop_linear_index)
  // size_t index;
  /// Starting cell for ParticleLoop called cell wise (intended for internal
  /// use).
  int starting_cell;
  /**
   * @returns The local linear index of the particle on this MPI rank in the
   * ParticleGroup
   */
  inline INT get_local_linear_index() const {
    return this->npart_cell_es[this->cell] + this->layer;
  }
  /**
   * @returns The linear index of the particle within the current ParticleLoop.
   */
  inline INT get_loop_linear_index() const {
    const INT linear_index =
        (this->npart_cell_es_lb[this->cell] + this->loop_layer) -
        this->npart_cell_es_lb[this->starting_cell];
    return linear_index;
  }
  /**
   * @returns The local linear index of the particle in the ParticleSubGroup.
   * This call is identical to get_local_linear_index when the iteration set is
   * a ParticleGroup.
   */
  inline INT get_sub_linear_index() const {
    return this->npart_cell_es_lb[this->cell] + this->loop_layer;
  }
};

} // namespace Access::LoopIndex

namespace ParticleLoopImplementation {

/**
 *  KernelParameter type for read-only access to a ParticleLoopIndex.
 */
template <> struct KernelParameter<Access::Read<ParticleLoopIndex>> {
  using type = Access::LoopIndex::Read;
};

/**
 * Host to loop type for index.
 */
struct ParticleLoopIndexKernelT {
  int starting_cell;
  int loop_type_int;
  INT const *npart_cell_es;
  INT const *npart_cell_es_lb;
};

/**
 *  Loop parameter for read access of a ParticleLoopIndex.
 */
template <> struct LoopParameter<Access::Read<ParticleLoopIndex>> {
  using type = ParticleLoopIndexKernelT;
};
/**
 * Method to compute access to a ParticleLoopIndex (read)
 */
inline ParticleLoopIndexKernelT
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,[[maybe_unused]]  Access::Read<ParticleLoopIndex *> &a) {
  NESOASSERT(global_info->loop_type_int == 0 || global_info->loop_type_int == 1,
             "Unknown loop type for ParticleLoopIndex.");
  ParticleLoopIndexKernelT tmp;
  tmp.starting_cell = global_info->starting_cell;
  tmp.loop_type_int = global_info->loop_type_int;
  tmp.npart_cell_es = global_info->d_npart_cell_es;
  tmp.npart_cell_es_lb = global_info->d_npart_cell_es_lb;
  return tmp;
}
/**
 *  Function to create the kernel argument for ParticleLoopIndex read access.
 */
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              ParticleLoopIndexKernelT &rhs,
                              Access::LoopIndex::Read &lhs) {
  lhs.cell = iterationx.cellx;
  lhs.layer = iterationx.layerx;
  lhs.loop_layer = iterationx.loop_layerx;
  // lhs.index = iterationx.index;
  lhs.starting_cell = rhs.starting_cell;
  lhs.loop_type_int = rhs.loop_type_int;
  lhs.npart_cell_es = rhs.npart_cell_es;
  lhs.npart_cell_es_lb = rhs.npart_cell_es_lb;
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles
#endif
