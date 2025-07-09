#ifndef _NESO_PARTICLES_ALGORITHMS_PARTICLE_DATA_MOVEMENT_HPP_
#define _NESO_PARTICLES_ALGORITHMS_PARTICLE_DATA_MOVEMENT_HPP_

#include "../compute_target.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

/**
 * Copy all particle data from an EphemeralDat to a ParticleDat. The source and
 * destination dats must exist and have the same number of components. Source
 * and destination dats may share the same name.
 *
 * @param particle_sub_group ParticleSubGroup containing source and destination
 * dats.
 * @param sym_src Source EphemeralDat name.
 * @param sym_dst Destination ParticleDat name.
 */
template <typename T>
void copy_ephemeral_dat_to_particle_dat(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<T> sym_src,
    Sym<T> sym_dst) {

  auto particle_group = get_particle_group(particle_sub_group);
  NESOASSERT(particle_sub_group->contains_ephemeral_dat(sym_src),
             "ParticleSubGroup does not contain an EphemeralDat with name: " +
                 sym_src.name);
  NESOASSERT(particle_group->contains_dat(sym_dst),
             "ParticleGroup does not contain a ParticleDat with name: " +
                 sym_dst.name);

  auto dat_src = particle_sub_group->get_ephemeral_dat(sym_src);
  auto dat_dst = particle_group->get_dat(sym_dst);

  NESOASSERT(
      dat_src->ncomp == dat_dst->ncomp,
      "Missmatch between number of components in source and destination.");

  const int k_ncomp = dat_src->ncomp;
  particle_loop(
      "copy_ephemeral_dat_to_particle_dat", particle_sub_group,
      [=](auto SRC, auto DST) {
        for (int cx = 0; cx < k_ncomp; cx++) {
          DST.at(cx) = SRC.at_ephemeral(cx);
        }
      },
      Access::read(dat_src), Access::write(dat_dst))
      ->execute();
}

extern template void
copy_ephemeral_dat_to_particle_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<INT> sym_src, Sym<INT> sym_dst);
extern template void
copy_ephemeral_dat_to_particle_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<REAL> sym_src, Sym<REAL> sym_dst);

/**
 * Copy all particle data from a ParticleDat to an EphemeralDat. The source and
 * destination dats must exist and have the same number of components. Source
 * and destination dats may share the same name.
 *
 * @param particle_sub_group ParticleSubGroup containing source and destination
 * dats.
 * @param sym_src Source ParticleDat name.
 * @param sym_dst Destination EphemeralDat name.
 */
template <typename T>
void copy_particle_dat_to_ephemeral_dat(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<T> sym_src,
    Sym<T> sym_dst) {

  auto particle_group = get_particle_group(particle_sub_group);
  NESOASSERT(particle_sub_group->contains_ephemeral_dat(sym_dst),
             "ParticleSubGroup does not contain an EphemeralDat with name: " +
                 sym_dst.name);
  NESOASSERT(particle_group->contains_dat(sym_src),
             "ParticleGroup does not contain a ParticleDat with name: " +
                 sym_src.name);

  auto dat_src = particle_group->get_dat(sym_src);
  auto dat_dst = particle_sub_group->get_ephemeral_dat(sym_dst);

  NESOASSERT(
      dat_src->ncomp == dat_dst->ncomp,
      "Missmatch between number of components in source and destination.");

  const int k_ncomp = dat_src->ncomp;
  particle_loop(
      "copy_particle_dat_to_ephemeral_dat", particle_sub_group,
      [=](auto SRC, auto DST) {
        for (int cx = 0; cx < k_ncomp; cx++) {
          DST.at_ephemeral(cx) = SRC.at(cx);
        }
      },
      Access::read(dat_src), Access::write(dat_dst))
      ->execute();
}

extern template void
copy_particle_dat_to_ephemeral_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<INT> sym_src, Sym<INT> sym_dst);
extern template void
copy_particle_dat_to_ephemeral_dat(ParticleSubGroupSharedPtr particle_sub_group,
                                   Sym<REAL> sym_src, Sym<REAL> sym_dst);

} // namespace NESO::Particles

#endif
