#include <neso_particles/pair_loop/pair_mask.hpp>

namespace NESO::Particles {

PairMask::PairMask(SYCLTargetSharedPtr sycl_target)
    : MaskArray(sycl_target, 1) {}

namespace ParticleLoopImplementation {

/**
 * Method to compute access to a PairMask (write)
 */
MaskArrayDevice
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<PairMask *> &a) {
  auto tmp = a.obj->get_device();
  return tmp;
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles
