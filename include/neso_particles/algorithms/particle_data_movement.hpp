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

/**
 * Set the specified component and property on all particles to the value in the
 * passed array at the index that corresponds to the cell of the particle.
 *
 * @param group ParticleGroup or ParticleSubGroup of particles to set values
 * for.
 * @param sym Particle property to set.
 * @param component Particle component to set.
 * @param values Vector of values to set cell wise.
 */
template <typename GROUP_TYPE, typename SYM_TYPE, typename VALUE_TYPE>
void cellwise_broadcast(std::shared_ptr<GROUP_TYPE> group, Sym<SYM_TYPE> sym,
                        const int component,
                        const std::vector<VALUE_TYPE> &values) {

  auto particle_group = get_particle_group(group);
  const int cell_count = particle_group->domain->mesh->get_cell_count();

  NESOASSERT((values.size() == static_cast<std::size_t>(cell_count)),
             "Number elements in values doesn't match the number of cells in "
             "domain.");

  auto sycl_target = particle_group->sycl_target;
  auto d_buffer = get_resource<BufferDevice<VALUE_TYPE>,
                               ResourceStackInterfaceBufferDevice<VALUE_TYPE>>(
      sycl_target->resource_stack_map,
      ResourceStackKeyBufferDevice<VALUE_TYPE>{}, sycl_target);
  d_buffer->realloc_no_copy(cell_count);

  auto *k_buffer = d_buffer->ptr;

  auto copy_event = sycl_target->queue.memcpy(k_buffer, values.data(),
                                              cell_count * sizeof(VALUE_TYPE));

  auto loop = particle_loop(
      "cellwise_broadcast", group,
      [=](auto INDEX, auto SYM) { SYM.at(component) = k_buffer[INDEX.cell]; },
      Access::read(ParticleLoopIndex{}), Access::write(sym));

  copy_event.wait();
  loop->execute();

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<VALUE_TYPE>{}, d_buffer);
}

/**
 * Fill a ParticleDat with the specified value.
 *
 * @param particle_group ParticleGroup containing ParticleDat to fill.
 * @param sym Sym<INT> or Sym<REAL> of ParticleDat to fill.
 * @param value INT or REAL value to fill ParticleDat with.
 * @param component Optionally specifify the component of the ParticleDat to
 * fill.
 */
template <typename T>
inline void fill(ParticleGroupSharedPtr particle_group, Sym<T> sym,
                 const T value, std::optional<int> component = std::nullopt) {
  NESOASSERT(particle_group->contains_dat(sym),
             "ParticleDat not found for sym with name: " + sym.name);

  if (component == std::nullopt) {
    const int ndim = particle_group->get_dat(sym)->ncomp;

    particle_loop(
        "Algorithms::fill", particle_group,
        [=](auto S) {
          for (int dx = 0; dx < ndim; dx++) {
            S.at(dx) = value;
          }
        },
        Access::write(sym))
        ->execute();

  } else {
    const int component_value = component.value();
    particle_loop(
        "Algorithms::fill", particle_group,
        [=](auto S) { S.at(component_value) = value; }, Access::write(sym))
        ->execute();
  }
}

extern template void fill(ParticleGroupSharedPtr, Sym<REAL>, const REAL,
                          std::optional<int>);
extern template void fill(ParticleGroupSharedPtr, Sym<INT>, const INT,
                          std::optional<int>);

/**
 * Fill a ParticleDat with the specified value.
 *
 * @param particle_sub_group ParticleSubGroup containing ParticleDat to fill.
 * @param sym Sym<INT> or Sym<REAL> of ParticleDat to fill.
 * @param value INT or REAL value to fill ParticleDat with.
 * @param component Optionally specifify the component of the ParticleDat to
 * fill.
 */
template <typename T>
inline void fill(ParticleSubGroupSharedPtr particle_sub_group, Sym<T> sym,
                 const T value, std::optional<int> component = std::nullopt) {

  const bool is_ephemeral = particle_sub_group->contains_ephemeral_dat(sym);
  auto particle_group = get_particle_group(particle_sub_group);
  NESOASSERT(particle_group->contains_dat(sym) || is_ephemeral,
             "ParticleDat not found for sym with name: " + sym.name);

  if (component == std::nullopt) {

    if (is_ephemeral) {
      const int ndim = particle_sub_group->get_ephemeral_dat(sym)->ncomp;
      particle_loop(
          "Algorithms::fill", particle_sub_group,
          [=](auto S) {
            for (int dx = 0; dx < ndim; dx++) {
              S.at_ephemeral(dx) = value;
            }
          },
          Access::write(sym))
          ->execute();

    } else {
      const int ndim = particle_group->get_dat(sym)->ncomp;
      particle_loop(
          "Algorithms::fill", particle_sub_group,
          [=](auto S) {
            for (int dx = 0; dx < ndim; dx++) {
              S.at(dx) = value;
            }
          },
          Access::write(sym))
          ->execute();
    }

  } else {
    const int component_value = component.value();

    if (is_ephemeral) {
      particle_loop(
          "Algorithms::fill", particle_sub_group,
          [=](auto S) { S.at_ephemeral(component_value) = value; },
          Access::write(sym))
          ->execute();

    } else {
      particle_loop(
          "Algorithms::fill", particle_sub_group,
          [=](auto S) { S.at(component_value) = value; }, Access::write(sym))
          ->execute();
    }
  }
}

extern template void fill(ParticleSubGroupSharedPtr, Sym<REAL>, const REAL,
                          std::optional<int>);
extern template void fill(ParticleSubGroupSharedPtr, Sym<INT>, const INT,
                          std::optional<int>);

} // namespace NESO::Particles

#endif
