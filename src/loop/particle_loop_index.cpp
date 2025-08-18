#include <neso_particles/loop/particle_loop_index.hpp>

namespace NESO::Particles {
namespace ParticleLoopImplementation {

ParticleLoopIndexKernelT
create_loop_arg(ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                [[maybe_unused]] Access::Read<ParticleLoopIndex *> &a) {
  NESOASSERT(global_info->loop_type_int == 0 || global_info->loop_type_int == 1,
             "Unknown loop type for ParticleLoopIndex.");
  ParticleLoopIndexKernelT tmp;
  tmp.starting_cell = global_info->starting_cell;
  tmp.loop_type_int = global_info->loop_type_int;
  tmp.npart_cell_es = global_info->d_npart_cell_es;
  tmp.npart_cell_es_lb = global_info->d_npart_cell_es_lb;
  return tmp;
}

} // namespace ParticleLoopImplementation
} // namespace NESO::Particles
