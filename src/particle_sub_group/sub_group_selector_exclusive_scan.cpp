#include <neso_particles/particle_sub_group/sub_group_selector_exclusive_scan.hpp>

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

template void SubGroupSelectorExclusiveScan::create_loop_1(
    std::shared_ptr<ParticleGroup> parent);
template void SubGroupSelectorExclusiveScan::create_loop_1(
    std::shared_ptr<ParticleSubGroup> parent);

SubGroupSelectorExclusiveScan::SubGroupSelectorExclusiveScan(
    std::shared_ptr<ParticleGroup> parent)
    : SubGroupSelectorBase(parent) {
  this->create_loop_1(parent);
}
SubGroupSelectorExclusiveScan::SubGroupSelectorExclusiveScan(
    std::shared_ptr<ParticleSubGroup> parent)
    : SubGroupSelectorBase(parent) {
  this->create_loop_1(parent);
}

void SubGroupSelectorExclusiveScan::create(Selection *created_selection) {
  return this->create_callback(created_selection);
}

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles
