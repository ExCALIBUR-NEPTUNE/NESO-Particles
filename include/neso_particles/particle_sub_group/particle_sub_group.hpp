#ifndef _PARTICLE_SUB_GROUP_H_
#define _PARTICLE_SUB_GROUP_H_
#include "../loop/particle_loop.hpp"
#include "../particle_group.hpp"
#include "../typedefs.hpp"
#include <type_traits>

#include "cell_sub_group_selector.hpp"
#include "particle_loop_sub_group.hpp"
#include "particle_sub_group_base.hpp"
#include "particle_sub_group_base_impl.hpp"
#include "sub_group_selector.hpp"
#include "sub_group_selector_base.hpp"
#include "sub_group_selector_base_impl.hpp"

namespace NESO::Particles {

/**
 * Create a ParticleSubGroup based on a kernel and arguments. The selector
 * kernel must be a lambda which returns true for particles which are in the
 * sub group and false for particles which are not in the sub group. The
 * arguments for the selector kernel must be read access Syms, i.e.
 * Access::read(Sym<T>("name")).
 *
 * For example if A is a ParticleGroup with an INT ParticleProp "ID" that
 * holds particle ids then the following line creates a ParticleSubGroup from
 * the particles with even ids.
 *
 *    auto A_even = std::make_shared<ParticleSubGroup>(
 *      A, [=](auto ID) {
 *        return ((ID[0] % 2) == 0);
 *      },
 *      Access::read(Sym<INT>("ID")));
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param kernel Lambda function (like a ParticleLoop kernel) that returns
 * true for the particles which should be in the ParticleSubGroup.
 * @param args Arguments in the form of access descriptors wrapping objects
 * to pass to the kernel.
 */
template <typename PARENT, typename KERNEL, typename... ARGS>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, KERNEL kernel,
                   ARGS... args) {
  return std::make_shared<ParticleSubGroup>(parent, kernel, args...);
}

/**
 * Create a ParticleSubGroup which is simply a reference/view into an entire
 * ParticleGroup. This constructor creates a sub-group which is equivalent to
 *
 *    auto A_all = std::make_shared<ParticleSubGroup>(
 *      A, [=]() {
 *        return true;
 *      }
 *    );
 *
 * but can make additional optimisations.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 */
template <typename PARENT>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent) {
  return std::make_shared<ParticleSubGroup>(parent);
}

/**
 * Create a static ParticleSubGroup based on a kernel and arguments. The
 * selector kernel must be a lambda which returns true for particles which are
 * in the sub group and false for particles which are not in the sub group. The
 * arguments for the selector kernel must be read access Syms, i.e.
 * Access::read(Sym<T>("name")).
 *
 * For example if A is a ParticleGroup with an INT ParticleProp "ID" that
 * holds particle ids then the following line creates a ParticleSubGroup from
 * the particles with even ids.
 *
 *    auto A_even = std::make_shared<ParticleSubGroup>(
 *      A, [=](auto ID) {
 *        return ((ID[0] % 2) == 0);
 *      },
 *      Access::read(Sym<INT>("ID")));
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param kernel Lambda function (like a ParticleLoop kernel) that returns
 * true for the particles which should be in the ParticleSubGroup.
 * @param args Arguments in the form of access descriptors wrapping objects
 * to pass to the kernel.
 */
template <typename PARENT, typename KERNEL, typename... ARGS>
inline ParticleSubGroupSharedPtr
static_particle_sub_group(std::shared_ptr<PARENT> parent, KERNEL kernel,
                          ARGS... args) {
  auto a = std::make_shared<ParticleSubGroup>(parent, kernel, args...);
  a->static_status(true);
  return a;
}

/**
 * Create a static ParticleSubGroup which is simply a reference/view into an
 * entire ParticleGroup. This constructor creates a sub-group which is
 * equivalent to
 *
 *    auto A_all = std::make_shared<ParticleSubGroup>(
 *      A, [=]() {
 *        return true;
 *      }
 *    );
 *
 * but can make additional optimisations.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 */
template <typename PARENT>
inline ParticleSubGroupSharedPtr
static_particle_sub_group(std::shared_ptr<PARENT> parent) {
  auto a = std::make_shared<ParticleSubGroup>(parent);
  a->static_status(true);
  return a;
}

/**
 * Create a ParticleSubGroup that selects all particles int a particular cell.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param cell Local cell index to select all particles in.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
template <typename PARENT, typename INT_TYPE,
          std::enable_if_t<std::is_integral<INT_TYPE>::value, bool> = true>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, const INT_TYPE cell,
                   const bool make_static = false) {
  auto selector = std::dynamic_pointer_cast<
      ParticleSubGroupImplementation::SubGroupSelector>(
      std::make_shared<ParticleSubGroupImplementation::CellSubGroupSelector>(
          parent, cell));
  auto group = std::make_shared<ParticleSubGroup>(selector);
  group->static_status(make_static);
  return group;
}

/**
 * Create a ParticleSubGroup that selects all particles int a particular cell.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param cell Local cell index to select all particles in.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
template <typename PARENT>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, const int cell,
                   const bool make_static = false) {
  auto selector = std::dynamic_pointer_cast<
      ParticleSubGroupImplementation::SubGroupSelector>(
      std::make_shared<ParticleSubGroupImplementation::CellSubGroupSelector>(
          parent, cell));
  auto group = std::make_shared<ParticleSubGroup>(selector);
  group->static_status(make_static);
  return group;
}

/**
 * Create a ParticleSubGroup that selects all particles int a particular cell.
 *
 * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
 * ParticleSubGroup.
 * @param cell Local cell index to select all particles in.
 * @param make_static Make the ParticleSubGroup static (default false).
 */
template <typename PARENT>
inline ParticleSubGroupSharedPtr
particle_sub_group(std::shared_ptr<PARENT> parent, const INT cell,
                   const bool make_static = false) {
  return particle_sub_group(parent, static_cast<int>(cell), make_static);
}

/**
 * Helper function to return the underlying ParticleGroup for a type.
 *
 * @param particle_sub_group ParticleSubGroup.
 * @returns Underlying ParticleGroup.
 */
inline auto get_particle_group(ParticleSubGroupSharedPtr particle_sub_group) {
  return particle_sub_group->get_particle_group();
}

/**
 * Helper function to return the underlying ParticleGroup for a type.
 *
 * @param particle_group ParticleGroup.
 * @returns Underlying ParticleGroup.
 */
inline auto get_particle_group(ParticleGroupSharedPtr particle_group) {
  return particle_group;
}

} // namespace NESO::Particles

#endif
