#ifndef _NESO_PARTICLES_PLI_PARTICLE_DAT_H_
#define _NESO_PARTICLES_PLI_PARTICLE_DAT_H_
#include "../loop/particle_loop_base.hpp"
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
  T const *const *ptr;
  /// Stores which particle in the cell this instance refers to.
  int layer;
  T const &operator[](const int component) {
    return ptr[component][this->layer];
  }
  T const &at(const int component) { return ptr[component][this->layer]; }
};

template <typename T> struct Write {
  /// Pointer to underlying data for a cell.
  T **ptr;
  /// Stores which particle in the cell this instance refers to.
  int layer;
  T &operator[](const int component) { return ptr[component][this->layer]; }
  T &at(const int component) { return ptr[component][this->layer]; }
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
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<Sym<T> *> &a) {
  auto sym = *a.obj;
  return global_info->particle_group->get_dat(sym)->impl_get_const();
}
/**
 * Method to compute access to a particle dat (write) - via Sym
 */
template <typename T>
inline ParticleDatImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<Sym<T> *> &a) {
  auto sym = *a.obj;
  return global_info->particle_group->get_dat(sym)->impl_get();
}
/**
 * Method to compute access to a particle dat (read).
 */
template <typename T>
inline ParticleDatImplGetConstT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<ParticleDatT<T> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a particle dat (write).
 */
template <typename T>
inline ParticleDatImplGetT<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<ParticleDatT<T> *> &a) {
  return a.obj->impl_get();
}

/**
 *  Function to create the kernel argument for ParticleDat read access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &IX, T *const *const *rhs,
                              Access::ParticleDat::Read<T> &lhs) {
  lhs.layer = IX.layerx;
  lhs.ptr = rhs[IX.cellx];
}
/**
 *  Function to create the kernel argument for ParticleDat write access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &IX, T ***rhs,
                              Access::ParticleDat::Write<T> &lhs) {
  lhs.layer = IX.layerx;
  lhs.ptr = rhs[IX.cellx];
}

} // namespace NESO::Particles::ParticleLoopImplementation

#endif
