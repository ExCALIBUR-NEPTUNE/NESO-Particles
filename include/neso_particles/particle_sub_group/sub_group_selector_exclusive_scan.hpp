#ifndef _NESO_PARTICLES_SUB_GROUP_SELECTOR_EXCLUSIVE_SCAN_HPP_
#define _NESO_PARTICLES_SUB_GROUP_SELECTOR_EXCLUSIVE_SCAN_HPP_

#include "../containers/local_array.hpp"
#include "../containers/sym_vector.hpp"
#include "../loop/cell_info_npart.hpp"
#include "../loop/particle_loop_functions.hpp"
#include "../loop/particle_loop_impl.hpp"
#include "../loop/particle_loop_index.hpp"
#include "particle_loop_sub_group_functions.hpp"
#include "particle_sub_group_utility.hpp"
#include "sub_group_selector_base.hpp"
#include "sub_group_selector_utility.hpp"

#include <functional>

namespace NESO::Particles {
namespace ParticleSubGroupImplementation {

/**
 * Class to consume the lambda which selects which particles are to be in a
 * ParticleSubGroup and provide to the ParticleSubGroup a list of cells and
 * layers. This class is functionally identical to SubGroupSelector but uses
 * exclusive scans instead of atomics.
 */
class SubGroupSelectorExclusiveScan : public SubGroupSelectorBase {
  friend class NESO::Particles::ParticleSubGroup;

protected:
  ParticleLoopSharedPtr loop_0;
  ParticleLoopSharedPtr loop_1;
  std::function<void(Selection *)> create_callback;

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

  SubGroupSelectorExclusiveScan() = default;

  template <typename T> inline void create_loop_1(std::shared_ptr<T> parent) {

    this->create_callback = [parent, this](Selection *created_selection) {
      auto sycl_target = get_particle_group(parent)->sycl_target;
      // std::tuple<int *, int *, INT *, INT *>
      auto [h_npart_cell_ptr, d_npart_cell_ptr, h_npart_cell_es_ptr,
            d_npart_cell_es_ptr_t] =
          this->sub_group_particle_map->get_helper_ptrs();
      // This purely avoids a compiler warning about capturing structured
      // bindings in lambdas.
      auto d_npart_cell_es_ptr = d_npart_cell_es_ptr_t;

      const int cell_count =
          this->particle_group->domain->mesh->get_cell_count();
      const INT npart_local = parent->get_npart_local();

      auto d_masks = get_resource<BufferDevice<int>,
                                  ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
      d_masks->realloc_no_copy(npart_local);
      int *k_masks = d_masks->ptr;

      auto e0 = sycl_target->queue.fill(static_cast<int *>(k_masks),
                                        static_cast<int>(0),
                                        static_cast<std::size_t>(npart_local));
      this->map_ptrs->set({k_masks, nullptr});
      e0.wait_and_throw();
      this->loop_0->submit();

      auto d_masks_es = get_resource<BufferDevice<int>,
                                     ResourceStackInterfaceBufferDevice<int>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
          sycl_target);
      d_masks_es->realloc_no_copy(npart_local);
      int *k_masks_es = d_masks_es->ptr;
      auto e1 = sycl_target->queue.fill(static_cast<int *>(k_masks_es),
                                        static_cast<int>(0),
                                        static_cast<std::size_t>(npart_local));

      auto d_npart_cell_es_parent =
          get_resource<BufferDevice<int>,
                       ResourceStackInterfaceBufferDevice<int>>(
              sycl_target->resource_stack_map,
              ResourceStackKeyBufferDevice<int>{}, sycl_target);
      d_npart_cell_es_parent->realloc_no_copy(cell_count);
      int *k_npart_cell_es_parent = d_npart_cell_es_parent->ptr;

      int *k_npart_cell_parent = Private::get_npart_cell_device_ptr(parent);
      INT *k_npart_cell_es_parent_INT =
          Private::get_npart_cell_es_device_ptr(parent);
      auto e2 = sycl_target->queue.parallel_for(
          sycl::range<1>(cell_count), [=](auto ix) {
            k_npart_cell_es_parent[ix] =
                static_cast<int>(k_npart_cell_es_parent_INT[ix]);
          });
      this->loop_0->wait();
      e1.wait_and_throw();
      e2.wait_and_throw();

      joint_exclusive_scan_n_sum(sycl_target,
                                 static_cast<std::size_t>(cell_count),
                                 k_npart_cell_parent, k_npart_cell_es_parent,
                                 k_masks, k_masks_es, d_npart_cell_ptr)
          .wait_and_throw();

      // Get the npart cell onto the host
      auto e3 = sycl_target->queue.memcpy(h_npart_cell_ptr, d_npart_cell_ptr,
                                          cell_count * sizeof(int));

      // Create the exclusive scan for the selection
      sycl_target->queue
          .fill(static_cast<int *>(k_npart_cell_es_parent), static_cast<int>(0),
                cell_count)
          .wait_and_throw();
      joint_exclusive_scan(sycl_target, cell_count, d_npart_cell_ptr,
                           k_npart_cell_es_parent)
          .wait_and_throw();
      // copy the exclusive scan for the selection into the correct place int ->
      // INT

      auto e4 = sycl_target->queue.parallel_for(
          sycl::range<1>(cell_count), [=](auto ix) {
            d_npart_cell_es_ptr[ix] = k_npart_cell_es_parent[ix];
          });

      auto e5 =
          sycl_target->queue.memcpy(h_npart_cell_es_ptr, d_npart_cell_es_ptr,
                                    cell_count * sizeof(INT), e4);

      e3.wait_and_throw();
      e5.wait_and_throw();
      this->sub_group_particle_map->create(0, cell_count, h_npart_cell_ptr,
                                           h_npart_cell_es_ptr);
      // Now the map exists and we can populate it
      auto d_cell_starts_ptr = this->sub_group_particle_map->d_cell_starts->ptr;

      particle_loop(
          "sub_group_selector_exclusive_scan_1", parent,
          [=](auto loop_index) {
            INT **base_map_cell_to_particles = d_cell_starts_ptr;
            const INT loop_linear_index = loop_index.get_loop_linear_index();
            const int required = k_masks[loop_linear_index];
            const int layer = k_masks_es[loop_linear_index];
            if (required) {
              INT *base_map_for_cell =
                  base_map_cell_to_particles[loop_index.cell];
              base_map_for_cell[layer] = loop_index.layer;
            }
          },
          Access::read(ParticleLoopIndex{}))
          ->execute();

      created_selection->npart_local =
          this->sub_group_particle_map->npart_total;
      created_selection->ncell = cell_count;
      created_selection->h_npart_cell = h_npart_cell_ptr;
      created_selection->d_npart_cell = d_npart_cell_ptr;
      created_selection->d_npart_cell_es = d_npart_cell_es_ptr;
      created_selection->d_map_cells_to_particles = {d_cell_starts_ptr};

      restore_resource(sycl_target->resource_stack_map,
                       ResourceStackKeyBufferDevice<int>{},
                       d_npart_cell_es_parent);
      restore_resource(sycl_target->resource_stack_map,
                       ResourceStackKeyBufferDevice<int>{}, d_masks);
      restore_resource(sycl_target->resource_stack_map,
                       ResourceStackKeyBufferDevice<int>{}, d_masks_es);
    };
  }

  SubGroupSelectorExclusiveScan(std::shared_ptr<ParticleGroup> parent);
  SubGroupSelectorExclusiveScan(std::shared_ptr<ParticleSubGroup> parent);

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
  SubGroupSelectorExclusiveScan(std::shared_ptr<PARENT> parent, KERNEL kernel,
                                ARGS... args)
      : SubGroupSelectorExclusiveScan(parent) {

    (this->check_read_access(args), ...);

    this->loop_0 = particle_loop(
        "sub_group_selector_exclusive_scan_0", parent,
        [=](auto loop_index, auto k_map_ptrs, auto... user_args) {
          const bool required = kernel(user_args...);
          const INT loop_linear_index = loop_index.get_loop_linear_index();
          if (required) {
            k_map_ptrs.at(0)[loop_linear_index] = 1;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(this->map_ptrs),
        args...);
  }

  virtual ~SubGroupSelectorExclusiveScan() = default;

  /**
   * Get two BufferDeviceHost objects that hold the cells and layers of the
   * particles which currently are selected by the selector kernel.
   *
   * @returns List of cells and layers of particles in the sub group.
   */
  virtual void create(Selection *created_selection) override;
};

extern template void SubGroupSelectorExclusiveScan::create_loop_1(
    std::shared_ptr<ParticleGroup> parent);
extern template void SubGroupSelectorExclusiveScan::create_loop_1(
    std::shared_ptr<ParticleSubGroup> parent);

typedef std::shared_ptr<SubGroupSelectorExclusiveScan>
    SubGroupSelectorExclusiveScanSharedPtr;

} // namespace ParticleSubGroupImplementation
} // namespace NESO::Particles

#endif
