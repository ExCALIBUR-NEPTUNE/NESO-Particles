#ifndef _NESO_PARTICLES_CONTAINERS_PARTICLE_SET_DEVICE_H_
#define _NESO_PARTICLES_CONTAINERS_PARTICLE_SET_DEVICE_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../particle_set.hpp"
#include "../particle_spec.hpp"
#include <map>
#include <memory>
#include <numeric>
#include <vector>

namespace NESO::Particles {

class ParticleSetDevice;

struct ParticleSetDeviceGet {
  REAL *ptr_real;
  INT *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_particles;
};

struct ParticleSetDeviceGetConst {
  REAL const *ptr_real;
  INT const *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_particles;
};

/**
 *  Defines the access implementations and types for ParticleSetDevice objects.
 */
namespace Access::ParticleSetDevice {

/**
 * Access:ParticleSetDevice::Read<T>, Access::ParticleSetDevice::Write<T> and
 * Access:ParticleSetDevice::Add<T> are the kernel argument types for accessing
 * ParticleSetDevice data in a kernel.
 */
/**
 * ParticleLoop access type for ParticleSetDevice Read access.
 */
struct Read {
  Read() = default;
  REAL const *ptr_real;
  INT const *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_particles;
  /**
   * Access a REAL particle property.
   *
   * @param particle The index of the particle to access, i.e. the row in the
   * device particle set.
   * @param property The particle property to access using the ordering REAL
   * ParticleProp instances were passed to the ParticleSetDeviceSpec
   * @param component The component of the property to access.
   * @returns Constant reference to particle value.
   */
  const REAL &at_real(const int particle, const int property,
                      const int component) const {
    return ptr_real[(offsets_real[property] + component) * num_particles +
                    particle];
  }
  /**
   * Access a INT particle property.
   *
   * @param particle The index of the particle to access, i.e. the row in the
   * device particle set.
   * @param property The particle property to access using the ordering INT
   * ParticleProp instances were passed to the ParticleSetDeviceSpec
   * @param component The component of the property to access.
   * @returns Constant reference to particle value.
   */
  const INT &at_int(const int particle, const int property,
                    const int component) const {
    return ptr_int[(offsets_int[property] + component) * num_particles +
                   particle];
  }
};

/**
 * ParticleLoop access type for ParticleSetDevice Add access.
 */
struct Add {
  Add() = default;
  REAL *ptr_real;
  INT *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_particles;

  /**
   * Atomically increment a REAL particle property.
   *
   * @param particle The index of the particle to access, i.e. the row in the
   * device particle set.
   * @param property The particle property to access using the ordering REAL
   * ParticleProp instances were passed to the ParticleSetDeviceSpec
   * @param component The component of the property to access.
   * @param value Value to increment property value by.
   * @returns Value prior to increment.
   */
  inline REAL fetch_add_real(const int particle, const int property,
                             const int component, const REAL value) {
    return atomic_fetch_add(
        &ptr_real[(offsets_real[property] + component) * num_particles +
                  particle],
        value);
  }

  /**
   * Atomically increment a INT particle property.
   *
   * @param particle The index of the particle to access, i.e. the row in the
   * device particle set.
   * @param property The particle property to access using the ordering INT
   * ParticleProp instances were passed to the ParticleSetDeviceSpec
   * @param component The component of the property to access.
   * @param value Value to increment property value by.
   * @returns Value prior to increment.
   */
  inline INT fetch_add_int(const int particle, const int property,
                           const int component, const INT value) {
    return atomic_fetch_add(
        &ptr_int[(offsets_int[property] + component) * num_particles +
                 particle],
        value);
  }
};

/**
 * ParticleLoop access type for ParticleSetDevice Write access.
 */
struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  REAL *ptr_real;
  INT *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_particles;

  /**
   * Write access to REAL particle property.
   *
   * @param particle The index of the particle to access, i.e. the row in the
   * device particle set.
   * @param property The particle property to access using the ordering REAL
   * ParticleProp instances were passed to the ParticleSetDeviceSpec
   * @param component The component of the property to access.
   * @returns Modifiable reference to property value.
   */
  REAL &at_real(const int particle, const int property, const int component) {
    return ptr_real[(offsets_real[property] + component) * num_particles +
                    particle];
  }

  /**
   * Write access to INT particle property.
   *
   * @param particle The index of the particle to access, i.e. the row in the
   * device particle set.
   * @param property The particle property to access using the ordering INT
   * ParticleProp instances were passed to the ParticleSetDeviceSpec
   * @param component The component of the property to access.
   * @returns Modifiable reference to property value.
   */
  INT &at_int(const int particle, const int property, const int component) {
    return ptr_int[(offsets_int[property] + component) * num_particles +
                   particle];
  }
};

} // namespace Access::ParticleSetDevice

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a ParticleSetDevice.
 */
template <> struct LoopParameter<Access::Read<ParticleSetDevice>> {
  using type = ParticleSetDeviceGetConst;
};
/**
 *  Loop parameter for write access of a ParticleSetDevice.
 */
template <> struct LoopParameter<Access::Write<ParticleSetDevice>> {
  using type = ParticleSetDeviceGet;
};
/**
 *  Loop parameter for add access of a ParticleSetDevice.
 */
template <> struct LoopParameter<Access::Add<ParticleSetDevice>> {
  using type = ParticleSetDeviceGet;
};
/**
 *  KernelParameter type for read access to a ParticleSetDevice.
 */
template <> struct KernelParameter<Access::Read<ParticleSetDevice>> {
  using type = Access::ParticleSetDevice::Read;
};
/**
 *  KernelParameter type for write access to a ParticleSetDevice.
 */
template <> struct KernelParameter<Access::Write<ParticleSetDevice>> {
  using type = Access::ParticleSetDevice::Write;
};
/**
 *  KernelParameter type for add access to a ParticleSetDevice.
 */
template <> struct KernelParameter<Access::Add<ParticleSetDevice>> {
  using type = Access::ParticleSetDevice::Add;
};
/**
 *  Function to create the kernel argument for ParticleSetDevice read access.
 */
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  ParticleSetDeviceGetConst &rhs,
                  Access::ParticleSetDevice::Read &lhs) {
  lhs.ptr_real = rhs.ptr_real;
  lhs.ptr_int = rhs.ptr_int;
  lhs.offsets_real = rhs.offsets_real;
  lhs.offsets_int = rhs.offsets_int;
  lhs.num_particles = rhs.num_particles;
}
/**
 *  Function to create the kernel argument for ParticleSetDevice write/add
 * access.
 */
template <typename T>
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                  ParticleSetDeviceGet &rhs, T &lhs) {
  lhs.ptr_real = rhs.ptr_real;
  lhs.ptr_int = rhs.ptr_int;
  lhs.offsets_real = rhs.offsets_real;
  lhs.offsets_int = rhs.offsets_int;
  lhs.num_particles = rhs.num_particles;
}

/**
 * Method to compute access to a ParticleSetDevice (read)
 */
inline ParticleSetDeviceGetConst
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<ParticleSetDevice *> &a);
/**
 * Method to compute access to a ParticleSetDevice (write)
 */
inline ParticleSetDeviceGet
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<ParticleSetDevice *> &a);
/**
 * Method to compute access to a ParticleSetDevice (add)
 */
inline ParticleSetDeviceGet
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Add<ParticleSetDevice *> &a);

} // namespace ParticleLoopImplementation

/**
 * Type to describe the particle properties of particles.
 */
struct ParticleSetDeviceSpec {
  /// The total number of components of type REAL stored.
  int num_components_real;
  /// The total number of components of type INT stored.
  int num_components_int;
  /// The number of REAL particle properties
  int num_properties_real;
  /// The number of INT particle properties
  int num_properties_int;
  /// The vector of REAL Syms.
  std::vector<Sym<REAL>> syms_real;
  /// The vector of INT Syms.
  std::vector<Sym<INT>> syms_int;
  /// The number of components per output property for REAL
  std::vector<int> components_real;
  /// The number of components per output property for INT
  std::vector<int> components_int;
  /// Default values applied on call to reset for REAL.
  std::map<std::pair<Sym<REAL>, int>, REAL> default_values_real;
  /// Default values applied on call to reset for INT.
  std::map<std::pair<Sym<INT>, int>, INT> default_values_int;
  /// Map from sym to integer index.
  std::map<Sym<REAL>, int> map_sym_index_real;
  /// Map from sym to integer index.
  std::map<Sym<INT>, int> map_sym_index_int;

  /// The ParticleSpec the instance was created from.
  ParticleSpec particle_spec;

  ParticleSetDeviceSpec() = default;
  ParticleSetDeviceSpec &operator=(const ParticleSetDeviceSpec &) = default;

  /**
   * Create a specification for particles based on a ParticleSpec.
   *
   * @param particle_spec Specification for particle particle properties.
   */
  ParticleSetDeviceSpec(ParticleSpec &particle_spec);

  /**
   * Set the default value for a particle property.
   *
   * @param sym Sym of particle property.
   * @param component Component of particle property.
   * @param value Default value to set.
   */
  void set_default_value(Sym<INT> sym, const int component, const INT value);

  /**
   * Set the default value for a particle property.
   *
   * @param sym Sym of particle property.
   * @param component Component of particle property.
   * @param value Default value to set.
   */
  void set_default_value(Sym<REAL> sym, const int component, const REAL value);

  /**
   * @returns the integer index that corresponds to a Sym in the specification.
   * Returns -1 if the Sym is not found.
   */
  int get_sym_index(const Sym<REAL> sym) const;

  /**
   * @returns the integer index that corresponds to a Sym in the specification.
   * Returns -1 if the Sym is not found.
   */
  int get_sym_index(const Sym<INT> sym) const;

  /**
   * @returns the number of components this spec stores for a Sym.
   */
  int get_num_components(const Sym<REAL> sym);

  /**
   * @returns the number of components this spec stores for a Sym.
   */
  int get_num_components(const Sym<INT> sym);
};

/**
 * Type to store N particles which can be passed to a ParticleLoop.
 * Fundamentally this class allocates two matrices, one for REAL valued
 * properties and one for INT value properties. These matrices are allocated
 * column major. Each output particle populates a row in these two matrices.
 * The column ordering is based on the ordering of properties and there
 * components in the input particle specification.
 */
class ParticleSetDevice {
  friend class ParticleGroup;
  friend inline ParticleSetDeviceGetConst
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<ParticleSetDevice *> &a);
  friend inline ParticleSetDeviceGet
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<ParticleSetDevice *> &a);
  friend inline ParticleSetDeviceGet
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<ParticleSetDevice *> &a);

protected:
  std::shared_ptr<BufferDevice<REAL>> d_data_real;
  std::shared_ptr<BufferDevice<INT>> d_data_int;
  std::shared_ptr<BufferDeviceHost<int>> dh_offsets_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_offsets_int;

  inline ParticleSetDeviceGet impl_get() {
    return {this->d_data_real->ptr, this->d_data_int->ptr,
            this->dh_offsets_real->d_buffer.ptr,
            this->dh_offsets_int->d_buffer.ptr, this->num_particles};
  }
  inline ParticleSetDeviceGetConst impl_get_const() {
    return {this->d_data_real->ptr, this->d_data_int->ptr,
            this->dh_offsets_real->d_buffer.ptr,
            this->dh_offsets_int->d_buffer.ptr, this->num_particles};
  }

public:
  /// The SYCLTarget particles are created on.
  SYCLTargetSharedPtr sycl_target;
  /// The number of particles stored.
  int num_particles;
  /// The specification of the particles.
  std::shared_ptr<ParticleSetDeviceSpec> spec;
  virtual ~ParticleSetDevice() = default;

  ParticleSetDevice() = default;

  /**
   * Note that the copy operator creates shallow copies of the array.
   */
  ParticleSetDevice &operator=(const ParticleSetDevice &) = default;

  /**
   * Create a new device particle set on the SYCLTarget. The reset method should
   * be called with the desired number of output particles before a loop is
   * executed which requires space for those output particles.
   *
   * @param sycl_target Device on which particle loops will be executed using
   * the device particle set.
   * @param spec A specification for the output particle properties.
   */
  ParticleSetDevice(SYCLTargetSharedPtr sycl_target,
                    std::shared_ptr<ParticleSetDeviceSpec> spec);

  /**
   * Create a new device particle set from a ParticleSec.
   *
   * @param sycl_target Device to store particles on.
   * @param num_particles Number of particles.
   * @param particle_spec ParticleSpec for particles.
   */
  ParticleSetDevice(SYCLTargetSharedPtr sycl_target, const int num_particles,
                    ParticleSpec &particle_spec);

  /**
   * Allocate space for a number of particle properties and fill the matrix
   * with the default values.
   *
   * @param reset Number of output particles to set in matrix.
   */
  virtual void reset(const int num_particles);

  /**
   * @returns a ParticleSet of the contained data.
   */
  ParticleSetSharedPtr get();

  /**
   * Set the data in the ParticleSetDevice from a ParticleSet.
   *
   * @param particle_set ParticleSet to set data from. Properties defined in the
   * ParticleSet which do not exist in the ParticleSetDevice are ignored.
   */
  void set(ParticleSetSharedPtr particle_set);
};

namespace ParticleLoopImplementation {
/**
 * Method to compute access to a ParticleSetDevice (read)
 */
inline ParticleSetDeviceGetConst
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ParticleSetDevice *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a ParticleSetDevice (write)
 */
inline ParticleSetDeviceGet
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ParticleSetDevice *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a ParticleSetDevice (add)
 */
inline ParticleSetDeviceGet
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Add<ParticleSetDevice *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
