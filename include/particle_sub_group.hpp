#ifndef _PARTICLE_SUB_GROUP_H_
#define _PARTICLE_SUB_GROUP_H_
#include "compute_target.hpp"
#include "loop/particle_loop.hpp"
#include "particle_group.hpp"
#include "typedefs.hpp"
#include <gtest/gtest.h>
#include <map>
#include <random>
#include <tuple>
#include <type_traits>

using namespace NESO::Particles;

namespace NESO::Particles {

namespace {

class SubGroupSelector {
protected:
  LocalArray<int> index;
  LocalArray<INT *> ptrs;
  ParticleLoopSharedPtr loop;

public:
  ParticleGroupSharedPtr particle_group;

  template <typename KERNEL, typename... ARGS>
  SubGroupSelector(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                   ARGS... args)
      : particle_group(particle_group) {

    auto sycl_target = particle_group->sycl_target;
    this->index = LocalArray<int>(sycl_target, 1);
    this->ptrs = LocalArray<INT *>(sycl_target, 2);

    this->loop = particle_loop(
        "sub_group_selector", particle_group,
        [=](auto loop_index, auto k_index, auto k_ptrs, auto... user_args) {
          const bool required = kernel(user_args...);
          if (required) {
            // increment the counter by 1 to get the index to store this
            // particle in
            const INT store_index = k_index(0, 1);
            INT *cells = k_ptrs[0];
            INT *layers = k_ptrs[1];
            cells[store_index] = loop_index.cell;
            layers[store_index] = loop_index.layer;
          }
        },
        Access::read(ParticleLoopIndex{}), Access::add(this->index),
        Access::read(this->ptrs), args...);
  }

  inline std::tuple<int, std::shared_ptr<BufferDeviceHost<INT>>,
                    std::shared_ptr<BufferDeviceHost<INT>>>
  get() {

    const int npart_local = particle_group->get_npart_local();
    auto sycl_target = this->particle_group->sycl_target;

    // TODO - cache these semi globally somewhere?
    auto ptr_cells =
        std::make_shared<BufferDevice<INT>>(sycl_target, npart_local);
    auto ptr_layers =
        std::make_shared<BufferDevice<INT>>(sycl_target, npart_local);

    std::vector<INT *> new_ptrs(2);
    new_ptrs[0] = ptr_cells->ptr;
    new_ptrs[1] = ptr_layers->ptr;
    this->ptrs.set(new_ptrs);

    this->index.fill(0);
    this->loop->execute();

    const int num_particles = this->index.get().at(0);
    auto cells_layers = this->ptrs.get();
    const INT *d_cells = cells_layers.at(0);
    const INT *d_layers = cells_layers.at(1);

    auto dh_cells =
        std::make_shared<BufferDeviceHost<INT>>(sycl_target, num_particles);
    auto dh_layers =
        std::make_shared<BufferDeviceHost<INT>>(sycl_target, num_particles);

    auto k_cells = dh_cells->d_buffer.ptr;
    auto k_layers = dh_layers->d_buffer.ptr;

    const std::size_t size = num_particles * sizeof(INT);
    auto e0 = sycl_target->queue.memcpy(k_cells, d_cells, size);
    auto e1 = sycl_target->queue.memcpy(k_layers, d_layers, size);
    e0.wait_and_throw();
    e1.wait_and_throw();
    dh_cells->device_to_host();
    dh_layers->device_to_host();

    return {num_particles, dh_cells, dh_layers};
  }
};

} // namespace

/**
 * TODO
 */
class ParticleSubGroup {
protected:
  ParticleGroupSharedPtr particle_group;
  SubGroupSelector selector;

  std::shared_ptr<BufferDeviceHost<INT>> dh_cells;
  std::shared_ptr<BufferDeviceHost<INT>> dh_layers;
  int npart_local;

  ParticleSubGroup(SubGroupSelector selector)
      : particle_group(selector.particle_group), selector(selector) {}

public:
  /**
   * TODO
   */
  template <typename KERNEL, typename... ARGS>
  ParticleSubGroup(ParticleGroupSharedPtr particle_group, KERNEL kernel,
                   ARGS... args)
      : ParticleSubGroup(SubGroupSelector(particle_group, kernel, args...)) {}

  inline void create() {
    auto buffers = this->selector.get();
    this->npart_local = std::get<0>(buffers);
    this->dh_cells = std::get<1>(buffers);
    this->dh_layers = std::get<2>(buffers);
  }
  
  /**
   * TODO - protect?
   */
  inline int get_cells_layers(std::vector<INT> &cells,
                              std::vector<INT> &layers) {
    cells.resize(npart_local);
    layers.resize(npart_local);
    for (int px = 0; px < npart_local; px++) {
      cells[px] = this->dh_cells->h_buffer.ptr[px];
      layers[px] = this->dh_layers->h_buffer.ptr[px];
    }
    return npart_local;
  }
};

} // namespace NESO::Particles

#endif
