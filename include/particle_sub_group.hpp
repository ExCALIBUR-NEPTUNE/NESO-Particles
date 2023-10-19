#ifndef _PARTICLE_SUB_GROUP_H_
#define _PARTICLE_SUB_GROUP_H_
#include "compute_target.hpp"
#include "loop/particle_loop.hpp"
#include "particle_group.hpp"
#include "typedefs.hpp"
#include <gtest/gtest.h>
#include <map>
#include <random>
#include <type_traits>

using namespace NESO::Particles;

namespace NESO::Particles {

class SubGroupSelector {
protected:
  ParticleGroupSharedPtr particle_group;
  LocalArray<int> index;
  LocalArray<INT *> ptrs;
  ParticleLoopSharedPtr loop;

  inline void resize() {
    static std::map<SYCLTargetSharedPtr, std::shared_ptr<BufferDevice<INT>>>
        cells;
    static std::map<SYCLTargetSharedPtr, std::shared_ptr<BufferDevice<INT>>>
        layers;

    const int npart_local = particle_group->get_npart_local();
    auto sycl_target = particle_group->sycl_target;

    auto ptr_cells = cells[sycl_target];
    if (!ptr_cells) {
      ptr_cells = std::make_shared<BufferDevice<INT>>(sycl_target, npart_local);
      cells[sycl_target] = ptr_cells;
    }
    auto ptr_layers = layers[sycl_target];
    if (!ptr_layers) {
      ptr_layers =
          std::make_shared<BufferDevice<INT>>(sycl_target, npart_local);
      cells[sycl_target] = ptr_layers;
    }

    ptr_cells->realloc_no_copy(npart_local);
    ptr_layers->realloc_no_copy(npart_local);

    // These are copied each time as another sub group selector may have
    // realloc'd the space
    std::vector<INT *> new_ptrs(2);
    new_ptrs[0] = ptr_cells->ptr;
    new_ptrs[1] = ptr_layers->ptr;
    this->ptrs.set(new_ptrs);
  }

public:
  template <typename KERNEL, typename... ARGS>
  SubGroupSelector(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                   ARGS... args)
      : particle_group(particle_group) {

    auto sycl_target = particle_group->sycl_target;
    this->index = LocalArray<int>(sycl_target, 1);
    this->ptrs = LocalArray<INT *>(sycl_target, 2);

    this->loop = particle_loop(
        "sub_group_selector", particle_group,
        [=](auto k_index, auto k_ptrs, auto... user_args) {
          const bool required = kernel(user_args...);
          if (required) {
            // increment the counter by 1 to get the index to store this
            // particle in
            const INT store_index = k_index(0, 1);
          }
        },
        Access::add(this->index), Access::read(this->index), args...);
  }

  inline void get() {
    this->resize();
    this->index.fill(0);
    this->loop->execute();
  }
};

class ParticleSubGroup {
protected:
  SubGroupSelector selector;

public:
  ParticleSubGroup(ParticleGroupSharedPtr particle_group,
                   SubGroupSelector selector)
      : selector(selector) {}
};

} // namespace NESO::Particles

#endif
