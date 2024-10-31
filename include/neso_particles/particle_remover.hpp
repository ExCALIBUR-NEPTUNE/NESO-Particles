#ifndef _NESO_PARTICLES_PARTICLE_REMOVER
#define _NESO_PARTICLES_PARTICLE_REMOVER

#include "compute_target.hpp"
#include "loop/particle_loop.hpp"
#include "particle_dat.hpp"
#include "particle_group.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  Utility to aid removing particles from a ParticleGroup based on a
 *  condition.
 */
class ParticleRemover {
private:
  BufferDeviceHost<int> dh_remove_count;
  BufferDevice<INT> d_remove_cells;
  BufferDevice<INT> d_remove_layers;
  SYCLTargetSharedPtr sycl_target;

public:
  /// Disable (implicit) copies.
  ParticleRemover(const ParticleRemover &st) = delete;
  /// Disable (implicit) copies.
  ParticleRemover &operator=(ParticleRemover const &a) = delete;

  /**
   *  Construct a remover that operates on ParticleGroups that use the given
   * SYCLTarget.
   *
   *  @param sycl_target SYCLTarget instance.
   */
  ParticleRemover(SYCLTargetSharedPtr sycl_target)
      : dh_remove_count(sycl_target, 1), d_remove_cells(sycl_target, 1),
        d_remove_layers(sycl_target, 1), sycl_target(sycl_target) {}

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
  template <typename T, typename U>
  inline void remove(ParticleGroupSharedPtr particle_group,
                     ParticleDatSharedPtr<T> particle_dat, const U key) {
    NESOASSERT(this->sycl_target == particle_group->sycl_target,
               "Passed ParticleGroup does not contain the sycl_target this "
               "ParticleRemover was created with.");

    auto k_key = key;

    // reset the leave count
    this->dh_remove_count.h_buffer.ptr[0] = 0;
    this->dh_remove_count.host_to_device();
    auto k_leave_count = this->dh_remove_count.d_buffer.ptr;

    ParticleLoop(
        "particle_remover_stage_0", particle_group,
        [=](auto compare_dat) {
          const T compare_value = compare_dat[0];
          // Is this particle is getting removed?
          if (compare_value == k_key) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                remove_count_atomic{k_leave_count[0]};
            remove_count_atomic.fetch_add(1);
          }
        },
        Access::read(particle_dat->sym))
        .execute();

    // read from the device the number of particles to remove
    this->dh_remove_count.device_to_host();
    const int remove_count = this->dh_remove_count.h_buffer.ptr[0];
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
      ParticleLoop(
          "particle_remover_stage_1", particle_group,
          [=](auto loop_index, auto compare_dat) {
            const INT cellx = loop_index.cell;
            const INT layerx = loop_index.layer;
            const T compare_value = compare_dat[0];
            // Is this particle is getting removed?
            if (compare_value == k_key) {
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device>
                  remove_count_atomic{k_leave_count[0]};
              const int index = remove_count_atomic.fetch_add(1);
              k_remove_cells[index] = cellx;
              k_remove_layers[index] = layerx;
            }
          },
          Access::read(ParticleLoopIndex{}), Access::read(particle_dat->sym))
          .execute();

      // remove the particles from the particle_group
      particle_group->remove_particles(remove_count, k_remove_cells,
                                       k_remove_layers);
    }
  }
};

} // namespace NESO::Particles

#endif
