#ifndef _NESO_PARTICLES_SYM_VECTOR_H_
#define _NESO_PARTICLES_SYM_VECTOR_H_
#include "../compute_target.hpp"
#include "../particle_group.hpp"
#include "../particle_spec.hpp"
#include <initializer_list>
#include <memory>
#include <vector>

namespace NESO::Particles {

// Forward declaration of ParticleLoop such that SymVector can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;

/**
 * Enables ParticleDats to be accessed in ParticleLoops with a number of
 * ParticleDats determined at runtime.
 */
template <typename T> class SymVector {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;

protected:
  std::vector<ParticleDatSharedPtr<T>> dats;
  std::shared_ptr<BufferDeviceHost<T ***>> dh_root_ptrs;
  std::shared_ptr<BufferDeviceHost<T *const *const *>> dh_root_const_ptrs;
  inline void create(SYCLTargetSharedPtr sycl_target) {
    const int num_dats = this->dats.size();
    this->dh_root_ptrs =
        std::make_shared<BufferDeviceHost<T ***>>(sycl_target, num_dats);
    this->dh_root_const_ptrs =
        std::make_shared<BufferDeviceHost<T *const *const *>>(sycl_target,
                                                              num_dats);
    for (int dx = 0; dx < num_dats; dx++) {
      this->dh_root_ptrs->h_buffer.ptr[dx] = this->dats.at(dx)->impl_get();
      this->dh_root_const_ptrs->h_buffer.ptr[dx] =
          this->dats.at(dx)->impl_get_const();
    }
    this->dh_root_ptrs->host_to_device();
    this->dh_root_const_ptrs->host_to_device();
  }

  inline T ****impl_get() {
    const int num_dats = this->dats.size();
    for (int dx = 0; dx < num_dats; dx++) {
      this->dats.at(dx)->impl_get();
    }
    return this->dh_root_ptrs->d_buffer.ptr;
  }
  inline T *const *const **impl_get_const() {
    const int num_dats = this->dats.size();
    for (int dx = 0; dx < num_dats; dx++) {
      this->dats.at(dx)->impl_get_const();
    }
    return this->dh_root_const_ptrs->d_buffer.ptr;
  }

public:
  SymVector() = default;
  SymVector<T> &operator=(const SymVector<T> &) = default;

  /**
   * Create a SymVector using a ParticleGroup and a std::vector of Syms.
   *
   * @param particle_group ParticleGroup to use.
   * @param syms Vector of Syms to use from particle_group.
   */
  SymVector(ParticleGroupSharedPtr particle_group, std::vector<Sym<T>> syms) {
    this->dats.clear();
    this->dats.reserve(syms.size());
    for (auto &sx : syms) {
      this->dats.push_back(particle_group->get_dat(sx));
    }
    this->create(particle_group->sycl_target);
  }

  /**
   * Create a SymVector using a ParticleGroup and an initialiser list of syms,
   * e.g.
   *
   *  SymVector(particle_group, {Sym<INT>("a"), Sym<INT>("b")});
   *
   * @param particle_group ParticleGroup to use.
   * @param syms Syms to use from particle_group.
   */
  SymVector(ParticleGroupSharedPtr particle_group,
            std::initializer_list<Sym<T>> syms) {
    this->dats.clear();
    this->dats.reserve(syms.size());
    for (auto &sx : syms) {
      this->dats.push_back(particle_group->get_dat(sx));
    }
    this->create(particle_group->sycl_target);
  }
};

/**
 * Helper function to create a SymVector.
 *
 * @param particle_group ParticleGroup to use.
 * @param syms Vector of Syms to use from particle_group.
 */
template <typename T>
std::shared_ptr<SymVector<T>> sym_vector(ParticleGroupSharedPtr particle_group,
                                         std::vector<Sym<T>> syms) {
  return std::make_shared<SymVector<T>>(particle_group, syms);
}

/**
 * Helper function to create a SymVector.
 *
 * @param particle_group ParticleGroup to use.
 * @param syms Syms to use from particle_group.
 */
template <typename T>
std::shared_ptr<SymVector<T>> sym_vector(ParticleGroupSharedPtr particle_group,
                                         std::initializer_list<Sym<T>> syms) {
  return std::make_shared<SymVector<T>>(particle_group, syms);
}

} // namespace NESO::Particles
#endif
