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
  ParticleSetDeviceSpec(ParticleSpec &particle_spec)
      : particle_spec(particle_spec) {

    this->num_properties_real = particle_spec.properties_real.size();
    this->num_properties_int = particle_spec.properties_int.size();
    this->components_real = std::vector<int>(num_properties_real);
    this->components_int = std::vector<int>(num_properties_int);
    this->syms_real = std::vector<Sym<REAL>>(num_properties_real);
    this->syms_int = std::vector<Sym<INT>>(num_properties_int);

    for (int px = 0; px < num_properties_real; px++) {
      this->components_real.at(px) = particle_spec.properties_real.at(px).ncomp;
      const auto sym = particle_spec.properties_real.at(px).sym;
      this->syms_real.at(px) = sym;
      this->map_sym_index_real[sym] = px;
    }
    for (int px = 0; px < num_properties_int; px++) {
      this->components_int.at(px) = particle_spec.properties_int.at(px).ncomp;
      const auto sym = particle_spec.properties_int.at(px).sym;
      this->syms_int.at(px) = sym;
      this->map_sym_index_int[sym] = px;
    }
    this->num_components_real = std::accumulate(this->components_real.begin(),
                                                this->components_real.end(), 0);
    this->num_components_int = std::accumulate(this->components_int.begin(),
                                               this->components_int.end(), 0);
  }

  /**
   * Set the default value for a particle property.
   *
   * @param sym Sym of particle property.
   * @param component Component of particle property.
   * @param value Default value to set.
   */
  inline void set_default_value(Sym<INT> sym, const int component,
                                const INT value) {
    this->default_values_int[{sym, component}] = value;
  }

  /**
   * Set the default value for a particle property.
   *
   * @param sym Sym of particle property.
   * @param component Component of particle property.
   * @param value Default value to set.
   */
  inline void set_default_value(Sym<REAL> sym, const int component,
                                const REAL value) {
    this->default_values_real[{sym, component}] = value;
  }

  /**
   * @returns the integer index that corresponds to a Sym in the specification.
   * Returns -1 if the Sym is not found.
   */
  inline int get_sym_index(const Sym<REAL> sym) const {
    auto it = this->map_sym_index_real.find(sym);
    if (it == this->map_sym_index_real.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

  /**
   * @returns the integer index that corresponds to a Sym in the specification.
   * Returns -1 if the Sym is not found.
   */
  inline int get_sym_index(const Sym<INT> sym) const {
    auto it = this->map_sym_index_int.find(sym);
    if (it == this->map_sym_index_int.end()) {
      return -1;
    } else {
      return it->second;
    }
  }

  /**
   * @returns the number of components this spec stores for a Sym.
   */
  inline int get_num_components(const Sym<REAL> sym) {
    const int index = this->get_sym_index(sym);
    if (index < 0) {
      return 0;
    } else {
      return this->components_real.at(index);
    }
  }

  /**
   * @returns the number of components this spec stores for a Sym.
   */
  inline int get_num_components(const Sym<INT> sym) {
    const int index = this->get_sym_index(sym);
    if (index < 0) {
      return 0;
    } else {
      return this->components_int.at(index);
    }
  }
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
                    std::shared_ptr<ParticleSetDeviceSpec> spec)
      : sycl_target(sycl_target), num_particles(0), spec(spec) {

    NESOASSERT(sycl_target != nullptr, "sycl_target is nullptr.");
    this->d_data_real = std::make_shared<BufferDevice<REAL>>(sycl_target, 1);
    this->d_data_int = std::make_shared<BufferDevice<INT>>(sycl_target, 1);

    // These offsets are functions of the properties and components not the
    // number of particles

    // REAL offsets
    std::vector<int> h_offsets_real(spec->components_real.size());
    std::exclusive_scan(spec->components_real.begin(),
                        spec->components_real.end(), h_offsets_real.begin(), 0);

    this->dh_offsets_real =
        std::make_shared<BufferDeviceHost<int>>(sycl_target, h_offsets_real);
    this->dh_offsets_real->host_to_device();

    // INT offsets
    std::vector<int> h_offsets_int(spec->components_int.size());
    std::exclusive_scan(spec->components_int.begin(),
                        spec->components_int.end(), h_offsets_int.begin(), 0);

    this->dh_offsets_int =
        std::make_shared<BufferDeviceHost<int>>(sycl_target, h_offsets_int);
    this->dh_offsets_int->host_to_device();
  }

  /**
   * Create a new device particle set from a ParticleSec.
   *
   * @param sycl_target Device to store particles on.
   * @param num_particles Number of particles.
   * @param particle_spec ParticleSpec for particles.
   */
  ParticleSetDevice(SYCLTargetSharedPtr sycl_target, const int num_particles,
                    ParticleSpec &particle_spec)
      : ParticleSetDevice(sycl_target, std::make_shared<ParticleSetDeviceSpec>(
                                           particle_spec)) {
    this->reset(num_particles);
  }

  /**
   * Allocate space for a number of particle properties and fill the matrix
   * with the default values.
   *
   * @param reset Number of output particles to set in matrix.
   */
  virtual inline void reset(const int num_particles) {
    const auto spec = this->spec.get();
    NESOASSERT(spec != nullptr, "ParticleSetDevice is not initialised.");
    NESOASSERT(num_particles >= 0,
               "A negative number of particles does not make sense.");
    this->num_particles = num_particles;
    if (num_particles > 0) {
      this->d_data_real->realloc_no_copy(num_particles *
                                         spec->num_components_real);
      this->d_data_int->realloc_no_copy(num_particles *
                                        spec->num_components_int);
      EventStack es;

      // reset the values to either 0 or the set default value
      for (int sx = 0; sx < spec->num_properties_real; sx++) {
        for (int cx = 0; cx < spec->components_real[sx]; cx++) {
          const std::pair<Sym<REAL>, int> key = {spec->syms_real.at(sx), cx};
          const REAL value = spec->default_values_real.count(key) > 0
                                 ? spec->default_values_real.at(key)
                                 : 0;
          const int sym_offset = this->dh_offsets_real->h_buffer.ptr[sx];
          REAL *column_start =
              this->d_data_real->ptr + (sym_offset + cx) * num_particles;
          es.push(this->sycl_target->queue.fill(column_start, value,
                                                num_particles));
        }
      }
      for (int sx = 0; sx < spec->num_properties_int; sx++) {
        for (int cx = 0; cx < spec->components_int[sx]; cx++) {
          const std::pair<Sym<INT>, int> key = {spec->syms_int.at(sx), cx};
          const INT value = spec->default_values_int.count(key) > 0
                                ? spec->default_values_int.at(key)
                                : 0;
          const int sym_offset = this->dh_offsets_int->h_buffer.ptr[sx];
          INT *column_start =
              this->d_data_int->ptr + (sym_offset + cx) * num_particles;
          es.push(this->sycl_target->queue.fill(column_start, value,
                                                num_particles));
        }
      }

      es.wait();
    }
  }

  /**
   * @returns a ParticleSet of the contained data.
   */
  inline ParticleSetSharedPtr get() {
    auto particle_set = std::make_shared<ParticleSet>(
        this->num_particles, this->spec->particle_spec);
    if (this->num_particles > 0) {
      EventStack es;

      for (int sx = 0; sx < spec->num_properties_real; sx++) {
        const int num_components = this->spec->components_real[sx];
        const auto sym = this->spec->syms_real[sx];
        const int sym_offset = this->dh_offsets_real->h_buffer.ptr[sx];
        auto ptr_dst = particle_set->get_ptr(sym, 0, 0);
        const auto ptr_src =
            this->d_data_real->ptr + sym_offset * this->num_particles;
        const std::size_t num_bytes =
            this->num_particles * num_components * sizeof(REAL);
        es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
      }
      for (int sx = 0; sx < spec->num_properties_int; sx++) {
        const int num_components = this->spec->components_int[sx];
        const auto sym = this->spec->syms_int[sx];
        const int sym_offset = this->dh_offsets_int->h_buffer.ptr[sx];
        auto ptr_dst = particle_set->get_ptr(sym, 0, 0);
        const auto ptr_src =
            this->d_data_int->ptr + sym_offset * this->num_particles;
        const std::size_t num_bytes =
            this->num_particles * num_components * sizeof(INT);
        es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
      }

      es.wait();
    }
    return particle_set;
  }

  /**
   * Set the data in the ParticleSetDevice from a ParticleSet.
   *
   * @param particle_set ParticleSet to set data from. Properties defined in the
   * ParticleSet which do not exist in the ParticleSetDevice are ignored.
   */
  inline void set(ParticleSetSharedPtr particle_set) {
    NESOASSERT(this->num_particles == particle_set->npart,
               "Missmatch in number of particles.");
    if (this->num_particles > 0) {
      EventStack es;
      for (int sx = 0; sx < spec->num_properties_real; sx++) {
        const auto sym = this->spec->syms_real[sx];
        if (particle_set->contains(sym)) {
          const int num_components = this->spec->components_real[sx];
          NESOASSERT(num_components == particle_set->ncomp_real.at(sym),
                     "Missmatch in number of components for sym with name: " +
                         sym.name);
          const int sym_offset = this->dh_offsets_real->h_buffer.ptr[sx];
          auto ptr_src = particle_set->get_ptr(sym, 0, 0);
          const auto ptr_dst =
              this->d_data_real->ptr + sym_offset * this->num_particles;
          const std::size_t num_bytes =
              this->num_particles * num_components * sizeof(REAL);
          es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
        }
      }
      for (int sx = 0; sx < spec->num_properties_int; sx++) {
        const auto sym = this->spec->syms_int[sx];
        if (particle_set->contains(sym)) {
          const int num_components = this->spec->components_int[sx];
          NESOASSERT(num_components == particle_set->ncomp_int.at(sym),
                     "Missmatch in number of components for sym with name: " +
                         sym.name);
          const int sym_offset = this->dh_offsets_int->h_buffer.ptr[sx];
          auto ptr_src = particle_set->get_ptr(sym, 0, 0);
          const auto ptr_dst =
              this->d_data_int->ptr + sym_offset * this->num_particles;
          const std::size_t num_bytes =
              this->num_particles * num_components * sizeof(INT);
          es.push(this->sycl_target->queue.memcpy(ptr_dst, ptr_src, num_bytes));
        }
      }

      es.wait();
    }
  }
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
