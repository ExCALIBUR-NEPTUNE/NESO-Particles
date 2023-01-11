#ifndef _NESO_PARTICLES_PARTICLE_REMOVER
#define _NESO_PARTICLES_PARTICLE_REMOVER

#include "compute_target.hpp"
#include "particle_dat.hpp"
#include "particle_group.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

class ParticleRemover {
private:
  BufferDeviceHost<INT> dh_remove_count;
  BufferDevice<INT> d_remove_cells;
  BufferDevice<INT> d_remove_layers;
  SYCLTargetSharedPtr sycl_target;

public:
  /// Disable (implicit) copies.
  ParticleRemover(const ParticleRemover &st) = delete;
  /// Disable (implicit) copies.
  ParticleRemover &operator=(ParticleRemover const &a) = delete;

  ParticleRemover(SYCLTargetSharedPtr sycl_target)
      : sycl_target(sycl_target), dh_remove_count(sycl_target, 1),
        d_remove_cells(sycl_target, 1), d_remove_layers(sycl_target, 1) {}

  /**
   * Remove particles from a ParticleGroup based on a value in a ParticleDat.
   * For each particle, compares the first value in the ParticleDat with the
   * passed Key. Removes the particle if the value matches the key.
   *
   * @param particle_group ParticleGroup to remove particles from.
   * @param particle_dat ParticleDat to inspect to determine if particles
   * should be removed.
   * @param key Key to compare particle values with for removal.
   */
  template <typename T>
  inline void remove(ParticleGroupSharedPtr particle_group,
                     ParticleDatSharedPtr<T> particle_dat, const T key) {

    auto pl_iter_range = particle_dat->get_particle_loop_iter_range();
    auto pl_stride = particle_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = particle_dat->get_particle_loop_npart_cell();

    auto k_compare_dat = particle_dat->cell_dat.device_ptr();
    auto k_key = key;

    // reset the leave count
    this->dh_remove_count.h_buffer.ptr[0] = 0;
    this->dh_remove_count.host_to_device();
    auto k_leave_count = this->dh_remove_count.d_buffer.ptr;

    this->sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                const T compare_value = k_compare_dat[cellx][0][layerx];
                // Is this particle is getting removed?
                if (compare_value == k_key) {
                  sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device>
                      remove_count_atomic{k_leave_count[0]};
                  remove_count_atomic.fetch_add(1);
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    // read from the device the number of particles to remove
    this->dh_remove_count.device_to_host();
    const INT remove_count = this->dh_remove_count.h_buffer.ptr[0];
    if (remove_count > 0) {
      // reset the leave count
      this->dh_remove_count.h_buffer.ptr[0] = 0;
      this->dh_remove_count.host_to_device();
      // space to store the cells/layers indices for the removal
      this->d_remove_cells.realloc_no_copy(remove_count);
      this->d_remove_layers.realloc_no_copy(remove_count);
      auto k_remove_cells = this->d_remove_cells.ptr;
      auto k_remove_layers = this->d_remove_layers.ptr;

      // assemble the remove indices/layers
      this->sycl_target->queue
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for<>(
                sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                  NESO_PARTICLES_KERNEL_START
                  const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                  const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                  const T compare_value = k_compare_dat[cellx][0][layerx];
                  // Is this particle is getting removed?
                  if (compare_value == k_key) {
                    sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device>
                        remove_count_atomic{k_leave_count[0]};
                    const INT index = remove_count_atomic.fetch_add(1);
                    k_remove_cells[index] = cellx;
                    k_remove_layers[index] = layerx;
                  }
                  NESO_PARTICLES_KERNEL_END
                });
          })
          .wait_and_throw();

      // remove the particles from the particle_group
      particle_group->remove_particles(static_cast<int>(remove_count),
                                       k_remove_cells, k_remove_layers);
    }
  }
};

} // namespace NESO::Particles

#endif
