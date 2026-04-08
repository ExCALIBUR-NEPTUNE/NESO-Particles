#include <neso_particles/pair_loop/pair_mask.hpp>

namespace NESO::Particles {

PairMask::PairMask(SYCLTargetSharedPtr sycl_target)
    : MaskArray(sycl_target, 1) {}

} // namespace NESO::Particles
