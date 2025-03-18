#ifndef _NESO_PARTICLES_PARTICLE_LOOP_CELL_INFO_NPART_HPP_
#define _NESO_PARTICLES_PARTICLE_LOOP_CELL_INFO_NPART_HPP_

#include "particle_loop_base.hpp"

namespace NESO::Particles {

/**
 * The type to pass to a ParticleLoop with read access to access the number of
 * particles in the cell for the iteration set passed to the ParticleLoop.
 */
struct CellInfoNPart {};

/**
 * Namespace for the kernel types of CellInfoNPart.
 */
namespace Access::CellInfoNPart {

/**
 * Kernel type for read access.
 */
struct Read {
  /// Number of particles in the cell (intended for internal use).
  INT npart_cell;

  /**
   * @returns The number of particles in the cell for the iteration set of the
   * ParticleLoop.
   */
  inline INT get() const { return this->npart_cell; }
};

} // namespace Access::CellInfoNPart

namespace ParticleLoopImplementation {

/**
 * Device copyable host loop type for CellInfoNPart.
 */
struct CellInfoNPartKernelT {
  int const *d_npart_cell_lb;
};

/**
 *  KernelParameter type for read-only access to a CellInfoNPart.
 */
template <> struct KernelParameter<Access::Read<CellInfoNPart>> {
  using type = Access::CellInfoNPart::Read;
};

/**
 *  Loop parameter for read access of a CellInfoNPart.
 */
template <> struct LoopParameter<Access::Read<CellInfoNPart>> {
  using type = CellInfoNPartKernelT;
};

/**
 * Method to compute access to a CellInfoNPart (read)
 */
inline CellInfoNPartKernelT
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                [[maybe_unused]] Access::Read<CellInfoNPart *> &a) {
  NESOASSERT(global_info->loop_type_int == 0 || global_info->loop_type_int == 1,
             "Unknown loop type for CellInfoNPart.");
  return {global_info->d_npart_cell_lb};
}

/**
 *  Function to create the kernel argument for CellInfoNPart read access.
 */
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              CellInfoNPartKernelT &rhs,
                              Access::CellInfoNPart::Read &lhs) {
  lhs.npart_cell = static_cast<INT>(rhs.d_npart_cell_lb[iterationx.cellx]);
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
