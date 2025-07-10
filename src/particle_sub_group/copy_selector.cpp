#include <neso_particles/particle_sub_group/copy_selector.hpp>
#include <neso_particles/particle_sub_group/particle_sub_group.hpp>

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {
void CopySelector::create(Selection *created_selection) {
  auto sycl_target = this->particle_group->sycl_target;
  this->parent->create_if_required();
  auto parent_selection = this->parent->get_selection();
  *created_selection = parent_selection;
}

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles
