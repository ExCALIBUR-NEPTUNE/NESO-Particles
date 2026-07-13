#include <neso_particles/pair_loop/pair_utility.hpp>
#include <neso_particles/particle_linear_index.hpp>

namespace NESO::Particles {

void mask_off_referenced_particles(
    CellwisePairListAbsolute<ParticleGroup, CellwisePairList> pair_list,
    ParticleMaskSharedPtr particle_mask) {

  NESOASSERT(particle_mask != nullptr, "particle_mask is nullptr");

  const bool a_is_b = pair_list.A == pair_list.B;
  NESOASSERT(a_is_b, "Only implemented for the case of A==B.");

  auto k_particle_linear_index = get_particle_linear_index_device(pair_list.A);
  auto k_particle_mask = particle_mask->get_device();

  NESOASSERT(k_particle_mask.d_masks != nullptr,
             "ParticleMask::reset not called.");

  particle_pair_loop(
      "mask_off_referenced_particles", pair_list,
      [=](auto INDEX_A, auto INDEX_B) {
        {
          const INT linear_index_a =
              k_particle_linear_index.get_local_linear_index(INDEX_A.cell,
                                                             INDEX_A.layer);
          k_particle_mask.set(linear_index_a, 0, false);
        }
        const INT linear_index_b =
            k_particle_linear_index.get_local_linear_index(INDEX_B.cell,
                                                           INDEX_B.layer);
        k_particle_mask.set(linear_index_b, 0, false);
      },
      Access::A(Access::read(ParticlePairLoopIndex{})),
      Access::B(Access::read(ParticlePairLoopIndex{})))
      ->execute();
}

} // namespace NESO::Particles
