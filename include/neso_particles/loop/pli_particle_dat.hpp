#ifndef _NESO_PARTICLES_PLI_PARTICLE_DAT_H_
#define _NESO_PARTICLES_PLI_PARTICLE_DAT_H_
#include "../loop/particle_loop_base.hpp"
#include "../pair_loop/particle_pair_loop_base.hpp"
#include "../particle_group.hpp"

/**
 *  Defines the access implementations and types for ParticleDat objects.
 */
namespace NESO::Particles::Access::ParticleDat {

/**
 * Access:ParticleDat::Read<T> and Access:ParticleDat::Read<T> are the
 * kernel argument types for accessing particle data in a kernel.
 */
template <typename T> struct Read {
  /// Pointer to underlying data for a cell.
  T const *RESTRICT const *RESTRICT ptr;
  /// The iteration information.
  ParticleLoopImplementation::ParticleLoopIteration const *iterationx;
  T const &operator[](const int component) const {
    return ptr[component][this->iterationx->layerx];
  }
  T const &at(const int component) const {
    return ptr[component][this->iterationx->layerx];
  }
  T const &at_ephemeral(const int component) const {
    return ptr[component][this->iterationx->loop_layerx];
  }
};

template <typename T> struct Write {
  /// Pointer to underlying data for a cell.
  T *RESTRICT const *RESTRICT ptr;
  /// The iteration information.
  ParticleLoopImplementation::ParticleLoopIteration const *iterationx;
  T &operator[](const int component) const {
    return ptr[component][this->iterationx->layerx];
  }
  T &at(const int component) const {
    return ptr[component][this->iterationx->layerx];
  }
  T &at_ephemeral(const int component) const {
    return ptr[component][this->iterationx->loop_layerx];
  }
};

} // namespace NESO::Particles::Access::ParticleDat

namespace NESO::Particles::ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a ParticleDat via Sym.
 */
template <typename T> struct LoopParameter<Access::Read<Sym<T>>> {
  using type = T *const *const *;
};
/**
 *  Loop parameter for write access of a ParticleDat via Sym.
 */
template <typename T> struct LoopParameter<Access::Write<Sym<T>>> {
  using type = T ***;
};
/**
 *  Loop parameter for read access of a ParticleDat.
 */
template <typename T> struct LoopParameter<Access::Read<ParticleDatT<T>>> {
  using type = T *const *const *;
};
/**
 *  Loop parameter for write access of a ParticleDat.
 */
template <typename T> struct LoopParameter<Access::Write<ParticleDatT<T>>> {
  using type = T ***;
};
/**
 *  KernelParameter type for read-only access to a ParticleDat - via Sym.
 */
template <typename T> struct KernelParameter<Access::Read<Sym<T>>> {
  using type = Access::ParticleDat::Read<T>;
};
/**
 *  KernelParameter type for write access to a ParticleDat - via Sym.
 */
template <typename T> struct KernelParameter<Access::Write<Sym<T>>> {
  using type = Access::ParticleDat::Write<T>;
};
/**
 *  KernelParameter type for read-only access to a ParticleDat.
 */
template <typename T> struct KernelParameter<Access::Read<ParticleDatT<T>>> {
  using type = Access::ParticleDat::Read<T>;
};
/**
 *  KernelParameter type for write access to a ParticleDat.
 */
template <typename T> struct KernelParameter<Access::Write<ParticleDatT<T>>> {
  using type = Access::ParticleDat::Write<T>;
};

/**
 * Method to compute access to a particle dat (read) - via Sym.
 */
template <typename T>
inline ParticleDatImplGetConstT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh, Access::Read<Sym<T> *> &a);
/**
 * Method to compute access to a particle dat (write) - via Sym
 */
template <typename T>
inline ParticleDatImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<Sym<T> *> &a);

/**
 * Method to compute access to a particle dat (read).
 */
ParticleDatImplGetConstT<REAL>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ParticleDatT<REAL> *> &a);
/**
 * Method to compute access to a particle dat (write).
 */
ParticleDatImplGetT<REAL>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ParticleDatT<REAL> *> &a);

/**
 * Method to compute access to a particle dat (read).
 */
ParticleDatImplGetConstT<INT>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ParticleDatT<INT> *> &a);

/**
 * Method to compute access to a particle dat (write).
 */
ParticleDatImplGetT<INT>
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ParticleDatT<INT> *> &a);

/**
 *  Function to create the kernel argument for ParticleDat read access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              T *const *const *rhs,
                              Access::ParticleDat::Read<T> &lhs) {
  lhs.iterationx = &iterationx;
  lhs.ptr = rhs[iterationx.cellx];
}
/**
 *  Function to create the kernel argument for ParticleDat write access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx, T ***rhs,
                              Access::ParticleDat::Write<T> &lhs) {
  lhs.iterationx = &iterationx;
  lhs.ptr = rhs[iterationx.cellx];
}

} // namespace NESO::Particles::ParticleLoopImplementation

namespace NESO::Particles {
namespace ParticlePairLoopImplementation {

/**
 *  Function to create the kernel argument for ParticleDat read access.
 */
template <typename T>
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopImplementation::ParticlePairLoopIteration
        &iteration_pair,
    ParticleLoopImplementation::ParticleLoopIteration &iteration_particle,
    T *const *const *rhs, Access::ParticleDat::Read<T> &lhs) {
  lhs.iterationx = &iteration_particle;
  lhs.ptr = rhs[iteration_particle.cellx];
}
/**
 *  Function to create the kernel argument for ParticleDat write access.
 */
template <typename T>
inline void create_kernel_arg(
    [[maybe_unused]] ParticlePairLoopImplementation::ParticlePairLoopIteration
        &iteration_pair,
    ParticleLoopImplementation::ParticleLoopIteration &iteration_particle,
    T ***rhs, Access::ParticleDat::Write<T> &lhs) {
  lhs.iterationx = &iteration_particle;
  lhs.ptr = rhs[iteration_particle.cellx];
}

} // namespace ParticlePairLoopImplementation
} // namespace NESO::Particles

#endif
