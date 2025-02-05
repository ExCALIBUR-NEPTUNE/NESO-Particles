#ifndef _NESO_PARTICLES_SYM_VECTOR_H_
#define _NESO_PARTICLES_SYM_VECTOR_H_
#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../loop/particle_loop_index.hpp"
#include "../particle_group.hpp"
#include "../particle_spec.hpp"
#include <initializer_list>
#include <memory>
#include <vector>

namespace NESO::Particles {

// Forward declaration of ParticleLoop such that SymVector can define
// ParticleLoop as a friend class.
template <typename KERNEL, typename... ARGS> class ParticleLoop;
template <typename T> using SymVectorImplGetT = T ****;
template <typename T> using SymVectorImplGetConstT = T *const *const **;

/**
 *  Defines the access implementations and types for ParticleDat objects.
 */
namespace Access::SymVector {

/**
 * Access:SymVector::Read<T> and Access:SymVector::Read<T> are the
 * kernel argument types for accessing particle data in a kernel via a
 * SymVector.
 */
template <typename T> struct Read {
  int cell;
  int layer;
  /// Pointer to underlying data.
  T *const *const **ptr;
  T const &at(const int dat_index, const int cell, const int layer,
              const int component) const {
    return ptr[dat_index][cell][component][layer];
  }
  T const &at(const int dat_index,
              const Access::LoopIndex::Read &particle_index,
              const int component) const {
    return ptr[dat_index][particle_index.cell][component][particle_index.layer];
  }
  T const &at(const int dat_index, const int component) const {
    return ptr[dat_index][this->cell][component][this->layer];
  }
};

template <typename T> struct Write {
  int cell;
  int layer;
  /// Pointer to underlying data.
  T ****ptr;
  T &at(const int dat_index, const int cell, const int layer,
        const int component) const {
    return ptr[dat_index][cell][component][layer];
  }
  T &at(const int dat_index, const Access::LoopIndex::Read &particle_index,
        const int component) const {
    return ptr[dat_index][particle_index.cell][component][particle_index.layer];
  }
  T &at(const int dat_index, const int component) const {
    return ptr[dat_index][this->cell][component][this->layer];
  }
};

} // namespace Access::SymVector

namespace ParticleLoopImplementation {
/**
 *  Loop parameter for read access of a SymVector.
 */
template <typename T> struct LoopParameter<Access::Read<SymVector<T>>> {
  using type = T *const *const **;
};
/**
 *  Loop parameter for write access of a SymVector.
 */
template <typename T> struct LoopParameter<Access::Write<SymVector<T>>> {
  using type = T ****;
};
/**
 *  KernelParameter type for read-only access to a SymVector.
 */
template <typename T> struct KernelParameter<Access::Read<SymVector<T>>> {
  using type = Access::SymVector::Read<T>;
};
/**
 *  KernelParameter type for write access to a SymVector.
 */
template <typename T> struct KernelParameter<Access::Write<SymVector<T>>> {
  using type = Access::SymVector::Write<T>;
};

/**
 * Method to compute access to a SymVector (read).
 */
template <typename T>
inline SymVectorImplGetConstT<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<SymVector<T> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a SymVector (write).
 */
template <typename T>
inline SymVectorImplGetT<T>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<SymVector<T> *> &a) {
  return a.obj->impl_get();
}

/**
 *  Function to create the kernel argument for SymVector read access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              T *const *const **rhs,
                              Access::SymVector::Read<T> &lhs) {
  lhs.cell = iterationx.cellx;
  lhs.layer = iterationx.layerx;
  lhs.ptr = rhs;
}
/**
 *  Function to create the kernel argument for SymVector write access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx, T ****rhs,
                              Access::SymVector::Write<T> &lhs) {
  lhs.cell = iterationx.cellx;
  lhs.layer = iterationx.layerx;
  lhs.ptr = rhs;
}

} // namespace ParticleLoopImplementation

/**
 * Enables ParticleDats to be accessed in ParticleLoops with a number of
 * ParticleDats determined at runtime.
 */
template <typename T> class SymVector {
  // This allows the ParticleLoop to access the implementation methods.
  template <typename KERNEL, typename... ARGS> friend class ParticleLoop;
  friend SymVectorImplGetConstT<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<SymVector<T> *> &a);
  friend SymVectorImplGetT<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<SymVector<T> *> &a);

protected:
  ParticleGroupSharedPtr particle_group;
  std::vector<Sym<T>> syms;
  /// The ParticleDats referenced by the SymVector.
  std::vector<ParticleDatSharedPtr<T>> dats;

  inline SymVectorImplGetT<T> impl_get() {
    return this->particle_group->sym_vector_pointer_cache_dispatch->get(
        this->syms);
  }
  inline SymVectorImplGetConstT<T> impl_get_const() {
    return this->particle_group->sym_vector_pointer_cache_dispatch->get_const(
        this->syms);
  }

  inline void create() {
    this->particle_group->sym_vector_pointer_cache_dispatch->create(this->syms);
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
  SymVector(ParticleGroupSharedPtr particle_group, std::vector<Sym<T>> syms)
      : particle_group(particle_group), syms(syms) {
    this->create();
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
            std::initializer_list<Sym<T>> syms)
      : particle_group(particle_group), syms(syms) {
    this->create();
  }

  /**
   * @returns The ParticleDats that form the SymVector.
   */
  inline std::vector<ParticleDatSharedPtr<T>> const &get_particle_dats() {
    if (this->syms.size() != this->dats.size()) {
      for (auto &sx : this->syms) {
        this->dats.push_back(this->particle_group->get_dat(sx));
      }
    }

    return this->dats;
  }
};

template <typename T> using SymVectorSharedPtr = std::shared_ptr<SymVector<T>>;

} // namespace NESO::Particles
#endif
