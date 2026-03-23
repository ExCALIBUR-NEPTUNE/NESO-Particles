#include <neso_particles/pair_loop/cellwise_pair_list.hpp>

namespace NESO::Particles {

bool CellwisePairList::validate_pair_list(
    [[maybe_unused]] SYCLTargetSharedPtr sycl_target) {

  return false;
}
} // namespace NESO::Particles
