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
  /// The type of the ParticleLoop.
  int loop_type_int;
  /// pointer to the exclusive sum of particle counts in each cell.
  INT const *npart_cell_es;
  /// Loop iteration index
  size_t index;
  /// Starting cell for ParticleLoop called cell wise.
  int starting_cell;
  /**
   * @returns The global linear index of the particle on this MPI rank in the
   * ParticleGroup
   */
  inline INT get_global_linear_index() {
    return this->npart_cell_es[this->cell] + this->layer;
  }
  /**
   * @returns The linear index of the particle within the current ParticleLoop.
   */
  inline INT get_loop_linear_index() {
    const INT linear_particle_group =
        (this->npart_cell_es[this->cell] + this->layer) -
        this->npart_cell_es[this->starting_cell];
    const INT linear_particle_sub_group = static_cast<INT>(index);
    return (loop_type_int == 0) ? linear_particle_group
                                : linear_particle_sub_group;
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
static inline ParticleLoopIndexKernelT
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                sycl::handler &cgh, Access::Read<ParticleLoopIndex *> &a) {
  NESOASSERT(global_info->loop_type_int == 0 || global_info->loop_type_int == 1,
             "Unknown loop type for ParticleLoopIndex.");
  ParticleLoopIndexKernelT tmp;
  tmp.starting_cell = global_info->starting_cell;
  tmp.loop_type_int = global_info->loop_type_int;
  tmp.npart_cell_es = global_info->d_npart_cell_es;
  return tmp;
}
/**
 *  Function to create the kernel argument for ParticleLoopIndex read access.
 */
inline void create_kernel_arg(const size_t index, const int cellx,
                              const int layerx, ParticleLoopIndexKernelT &rhs,
                              Access::LoopIndex::Read &lhs) {
  lhs.cell = cellx;
  lhs.layer = layerx;
  lhs.index = index;
  lhs.starting_cell = rhs.starting_cell;
  lhs.loop_type_int = rhs.loop_type_int;
  lhs.npart_cell_es = rhs.npart_cell_es;
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles
#endif
