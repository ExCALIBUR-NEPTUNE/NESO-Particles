#ifndef _NESO_PARTICLES_ALGORITHMS_UNSEEN_VALUE_EXTRACTOR_HPP_
#define _NESO_PARTICLES_ALGORITHMS_UNSEEN_VALUE_EXTRACTOR_HPP_

#include "../compute_target.hpp"
#include "../containers/blocked_binary_tree.hpp"
#include "../loop/particle_loop.hpp"
#include "../loop/particle_loop_impl.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/particle_loop_sub_group_functions.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

#include <set>

namespace NESO::Particles {

/**
 * Helper type for extracting previously unseen values from a collection of
 * particles.
 */
class UnseenValueExtractor {
protected:
  SYCLTargetSharedPtr sycl_target;

  struct NodeType {
    INT a;
  };

  std::shared_ptr<BlockedBinaryTree<INT, NodeType>> d_tree;

public:
  /**
   * Create new UnseenValueExtractor on given compute device.
   *
   * @param sycl_target Compute device.
   */
  UnseenValueExtractor(SYCLTargetSharedPtr sycl_target);

  /**
   * Extract unseen values from a ParticleGroup or ParticleSubGroup.
   *
   * @param group ParticleGroup or ParticleSubGroup to extract values from.
   * @param sym INT valued Sym to extract values from.
   * @param component Component in sym to extract value from.
   * @param is_ephemeral Should the sym be treated as an EphemeralDat.
   * @returns std::set<INT> of previously unseen values.
   */
  template <typename GROUP_TYPE>
  std::set<INT> extract(std::shared_ptr<GROUP_TYPE> group, Sym<INT> sym,
                        const int component, const bool is_ephemeral) {

    const std::size_t npart_local = group->get_npart_local();
    auto d_buffer = get_resource<BufferDevice<INT>,
                                 ResourceStackInterfaceBufferDevice<INT>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<INT>{},
        sycl_target);
    d_buffer->realloc_no_copy(npart_local);
    auto d_counter = get_resource<BufferDevice<int>,
                                  ResourceStackInterfaceBufferDevice<int>>(
        sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<int>{},
        sycl_target);
    d_counter->realloc_no_copy(1);

    INT *k_buffer = d_buffer->ptr;
    int *k_counter = d_counter->ptr;
    sycl_target->queue.fill(k_counter, (int)0, 1).wait_and_throw();

    auto *k_tree = this->d_tree->root;

    if (is_ephemeral) {
      particle_loop(
          "UnseenValueExtractor::extract", group,
          [=](auto SYM) {
            const INT value = SYM.at_ephemeral(component);

            bool in_tree = false;
            const NodeType *node = nullptr;

            if (k_tree != nullptr) {
              in_tree = k_tree->get(value, &node);
            }

            // If this value is unseen
            if (!in_tree) {
              const int index = atomic_fetch_add(k_counter, 1);
              k_buffer[index] = value;
            }
          },
          Access::read(sym))
          ->execute();

    } else {
      particle_loop(
          "UnseenValueExtractor::extract", group,
          [=](auto SYM) {
            const INT value = SYM.at(component);

            bool in_tree = false;
            const NodeType *node = nullptr;

            if (k_tree != nullptr) {
              in_tree = k_tree->get(value, &node);
            }

            // If this value is unseen
            if (!in_tree) {
              const int index = atomic_fetch_add(k_counter, 1);
              k_buffer[index] = value;
            }
          },
          Access::read(sym))
          ->execute();
    }

    int h_counter = 0;
    this->sycl_target->queue.memcpy(&h_counter, k_counter, sizeof(int))
        .wait_and_throw();

    std::vector<INT> h_buffer;
    if (h_counter > 0) {
      h_buffer.resize(h_counter);
      this->sycl_target->queue
          .memcpy(h_buffer.data(), k_buffer, h_counter * sizeof(INT))
          .wait_and_throw();
    }

    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<int>{}, d_counter);
    restore_resource(sycl_target->resource_stack_map,
                     ResourceStackKeyBufferDevice<INT>{}, d_buffer);

    std::set<INT> output_set;
    for (const INT &ix : h_buffer) {
      output_set.insert(ix);
    }

    for (const INT &ix : output_set) {
      this->d_tree->add(ix, {ix});
    }

    return output_set;
  }
};

extern template std::set<INT>
UnseenValueExtractor::extract(std::shared_ptr<ParticleGroup> group,
                              Sym<INT> sym, const int component,
                              const bool is_ephemeral);

extern template std::set<INT>
UnseenValueExtractor::extract(std::shared_ptr<ParticleSubGroup> group,
                              Sym<INT> sym, const int component,
                              const bool is_ephemeral);

} // namespace NESO::Particles

#endif
