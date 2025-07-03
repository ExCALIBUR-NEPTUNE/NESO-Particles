#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_HPP_

#include "../containers/local_array.hpp"
#include "../containers/sym_vector.hpp"
#include "../loop/cell_info_npart.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../loop/particle_loop_index.hpp"
#include "particle_loop_selector_ordering_functions.hpp"
#include "sub_group_selector_base.hpp"

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

/**
 * Class to consume the lambda which selects which particles are to be in a
 * ParticleSubGroup and provide to the ParticleSubGroup a list of cells and
 * layers.
 */
class SubGroupSelector : public SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  ParticleLoopSharedPtr loop_0;
  ParticleLoopSharedPtr loop_1;

  template <template <typename> typename T, typename U>
  inline void check_sym_type(T<U> arg) {
    static_assert(std::is_same<T<U>, Sym<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Sym type check failed.");

    // add this sym to the version checker signature
    this->add_sym_dependency(arg);
  }

  template <typename T> inline void check_sym_type(SymVectorSharedPtr<T> sv) {
    auto dats = sv->get_particle_dats();
    for (auto &dx : dats) {
      this->check_sym_type(dx->sym);
    }
  }

  inline void check_sym_type([[maybe_unused]] ParticleLoopIndex &) {}
  inline void check_sym_type([[maybe_unused]] CellInfoNPart &) {}

  template <template <typename> typename T, typename U>
  inline void check_read_access(T<U> arg) {
    static_assert(std::is_same<T<U>, Access::Read<U>>::value == true,
                  "Filtering lambda arguments must be read access particle "
                  "properties (Sym instances). Read access check failed.");
    check_sym_type(arg.obj);
  }

  SubGroupSelector() = default;

  SubGroupSelector(std::shared_ptr<ParticleGroup> parent);
  SubGroupSelector(std::shared_ptr<ParticleSubGroup> parent);

public:
  /**
   * Create a selector based on a kernel and arguments. The selector kernel
   * must be a lambda which returns true for particles which are in the sub
   * group and false for particles which are not in the sub group. The
   * arguments for the selector kernel must be read access Syms, i.e.
   * Access::read(Sym<T>("name")), ParticleLoopIndex or CellInfoNPart.
   *
   * @param parent Parent ParticleGroup or ParticleSubGroup from which to form
   * ParticleSubGroup.
   * @param kernel Lambda function (like a ParticleLoop kernel) that returns
   * true for the particles which should be in the ParticleSubGroup.
   * @param args Arguments in the form of access descriptors wrapping objects
   * to pass to the kernel.
   */
  template <typename PARENT, typename KERNEL, typename... ARGS>
  SubGroupSelector(std::shared_ptr<PARENT> parent, KERNEL kernel, ARGS... args)
      : SubGroupSelector(parent) {

    (this->check_read_access(args), ...);

    this->loop_0 = particle_loop(
        "sub_group_selector_0", parent,
        [=](auto loop_index, auto k_map_ptrs, auto... user_args) {
          const bool required = kernel(user_args...);
          const INT particle_linear_index = loop_index.get_local_linear_index();
          if (required) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                element_atomic(k_map_ptrs.at(1)[loop_index.cell]);
            const int layer = element_atomic.fetch_add(1);
            k_map_ptrs.at(0)[particle_linear_index] = layer;
          } else {
            k_map_ptrs.at(0)[particle_linear_index] = -1;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs),
        args...);
  }

  virtual ~SubGroupSelector() = default;
  /**
   * Get two BufferDeviceHost objects that hold the cells and layers of the
   * particles which currently are selected by the selector kernel.
   *
   * @returns List of cells and layers of particles in the sub group.
   */
  virtual void create(Selection *created_selection) override;
};

typedef std::shared_ptr<SubGroupSelector> SubGroupSelectorSharedPtr;

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
