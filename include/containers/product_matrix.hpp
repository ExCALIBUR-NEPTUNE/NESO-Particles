#ifndef _NESO_PARTICLES_PRODUCT_MATRIX_H_
#define _NESO_PARTICLES_PRODUCT_MATRIX_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../particle_spec.hpp"
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace NESO::Particles {

class ProductMatrix;

struct ProductMatrixGet {
  REAL *ptr_real;
  INT *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_products;
};

struct ProductMatrixGetConst {
  REAL const *ptr_real;
  INT const *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_products;
};

/**
 *  Defines the access implementations and types for ProductMatrix objects.
 */
namespace Access::ProductMatrix {

/**
 * Access:ProductMatrix::Read<T>, Access::ProductMatrix::Write<T> and
 * Access:ProductMatrix::Add<T> are the kernel argument types for accessing
 * ProductMatrix data in a kernel.
 */
/**
 * ParticleLoop access type for ProductMatrix Read access.
 */
struct Read {
  Read() = default;
  REAL const *ptr_real;
  INT const *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_products;
  const REAL &at_real(const int product, const int property,
                      const int component) const {
    return ptr_real[(offsets_real[property] + component) * num_products +
                    product];
  }
  const INT &at_int(const int product, const int property,
                    const int component) const {
    return ptr_int[(offsets_int[property] + component) * num_products +
                   product];
  }
};

/**
 * ParticleLoop access type for ProductMatrix Add access.
 */
struct Add {
  Add() = default;
  REAL *ptr_real;
  INT *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_products;
  inline REAL fetch_add_real(const int product, const int property,
                             const int component, const REAL value) {
    sycl::atomic_ref<REAL, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        element_atomic(
            ptr_real[(offsets_real[property] + component) * num_products +
                     product]);
    return element_atomic.fetch_add(value);
  }
  inline INT fetch_add_int(const int product, const int property,
                           const int component, const INT value) {
    sycl::atomic_ref<INT, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        element_atomic(
            ptr_int[(offsets_int[property] + component) * num_products +
                    product]);
    return element_atomic.fetch_add(value);
  }
};

/**
 * ParticleLoop access type for ProductMatrix Write access.
 */
struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  REAL *ptr_real;
  INT *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_products;
  REAL &at_real(const int product, const int property, const int component) {
    return ptr_real[(offsets_real[property] + component) * num_products +
                    product];
  }
  INT &at_int(const int product, const int property, const int component) {
    return ptr_int[(offsets_int[property] + component) * num_products +
                   product];
  }
};

} // namespace Access::ProductMatrix

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a ProductMatrix.
 */
template <> struct LoopParameter<Access::Read<ProductMatrix>> {
  using type = ProductMatrixGetConst;
};
/**
 *  Loop parameter for write access of a ProductMatrix.
 */
template <> struct LoopParameter<Access::Write<ProductMatrix>> {
  using type = ProductMatrixGet;
};
/**
 *  Loop parameter for add access of a ProductMatrix.
 */
template <> struct LoopParameter<Access::Add<ProductMatrix>> {
  using type = ProductMatrixGet;
};
/**
 *  KernelParameter type for read access to a ProductMatrix.
 */
template <> struct KernelParameter<Access::Read<ProductMatrix>> {
  using type = Access::ProductMatrix::Read;
};
/**
 *  KernelParameter type for write access to a ProductMatrix.
 */
template <> struct KernelParameter<Access::Write<ProductMatrix>> {
  using type = Access::ProductMatrix::Write;
};
/**
 *  KernelParameter type for add access to a ProductMatrix.
 */
template <> struct KernelParameter<Access::Add<ProductMatrix>> {
  using type = Access::ProductMatrix::Add;
};
/**
 *  Function to create the kernel argument for ProductMatrix read access.
 */
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              ProductMatrixGetConst &rhs,
                              Access::ProductMatrix::Read &lhs) {
  lhs.ptr_real = rhs.ptr_real;
  lhs.ptr_int = rhs.ptr_int;
  lhs.offsets_real = rhs.offsets_real;
  lhs.offsets_int = rhs.offsets_int;
  lhs.num_products = rhs.num_products;
}
/**
 *  Function to create the kernel argument for ProductMatrix write/add access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              ProductMatrixGet &rhs, T &lhs) {
  lhs.ptr_real = rhs.ptr_real;
  lhs.ptr_int = rhs.ptr_int;
  lhs.offsets_real = rhs.offsets_real;
  lhs.offsets_int = rhs.offsets_int;
  lhs.num_products = rhs.num_products;
}

/**
 * Method to compute access to a ProductMatrix (read)
 */
inline ProductMatrixGetConst
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<ProductMatrix *> &a);
/**
 * Method to compute access to a ProductMatrix (write)
 */
inline ProductMatrixGet create_loop_arg(ParticleLoopGlobalInfo *global_info,
                                        sycl::handler &cgh,
                                        Access::Write<ProductMatrix *> &a);
/**
 * Method to compute access to a ProductMatrix (add)
 */
inline ProductMatrixGet create_loop_arg(ParticleLoopGlobalInfo *global_info,
                                        sycl::handler &cgh,
                                        Access::Add<ProductMatrix *> &a);

} // namespace ParticleLoopImplementation

/**
 * Type to describe the particle properties of products.
 */
struct ProductMatrixSpec {
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

  ProductMatrixSpec() = default;
  ProductMatrixSpec &operator=(const ProductMatrixSpec &) = default;

  /**
   * Create a specification for products based on a ParticleSpec.
   *
   * @param particle_spec Specification for product particle properties.
   */
  ProductMatrixSpec(ParticleSpec &particle_spec)
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
};

/**
 * Helper function to create ProductMatrixSpec instances.
 *
 * @param particle_spec Specification for product particle properties.
 */
inline std::shared_ptr<ProductMatrixSpec>
product_matrix_spec(ParticleSpec particle_spec) {
  return std::make_shared<ProductMatrixSpec>(particle_spec);
}

class ProductMatrix {
  friend class ParticleGroup;
  friend inline ProductMatrixGetConst
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<ProductMatrix *> &a);
  friend inline ProductMatrixGet ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<ProductMatrix *> &a);
  friend inline ProductMatrixGet ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<ProductMatrix *> &a);

protected:
  std::shared_ptr<BufferDevice<REAL>> d_data_real;
  std::shared_ptr<BufferDevice<INT>> d_data_int;
  std::shared_ptr<BufferDeviceHost<int>> dh_offsets_real;
  std::shared_ptr<BufferDeviceHost<int>> dh_offsets_int;

  inline ProductMatrixGet impl_get() {
    return {this->d_data_real->ptr, this->d_data_int->ptr,
            this->dh_offsets_real->d_buffer.ptr,
            this->dh_offsets_int->d_buffer.ptr, this->num_products};
  }
  inline ProductMatrixGetConst impl_get_const() {
    return {this->d_data_real->ptr, this->d_data_int->ptr,
            this->dh_offsets_real->d_buffer.ptr,
            this->dh_offsets_int->d_buffer.ptr, this->num_products};
  }

public:
  /// The SYCLTarget products are created on.
  SYCLTargetSharedPtr sycl_target;
  /// The number of products stored.
  int num_products;
  std::shared_ptr<ProductMatrixSpec> spec;

  ProductMatrix() = default;

  /**
   * Note that the copy operator creates shallow copies of the array.
   */
  ProductMatrix &operator=(const ProductMatrix &) = default;

  /**
   * TODO
   */
  ProductMatrix(SYCLTargetSharedPtr sycl_target,
                std::shared_ptr<ProductMatrixSpec> spec)
      : sycl_target(sycl_target), spec(spec) {

    NESOASSERT(sycl_target != nullptr, "sycl_target is nullptr.");
    this->d_data_real = std::make_shared<BufferDevice<REAL>>(sycl_target, 1);
    this->d_data_int = std::make_shared<BufferDevice<INT>>(sycl_target, 1);

    // These offsets are functions of the properties and components not the
    // number of products

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
   * TODO
   */
  inline void reset(const int num_products) {
    const auto spec = this->spec.get();
    NESOASSERT(spec != nullptr, "ProductMatrix is not initialised.");
    this->num_products = num_products;
    this->d_data_real->realloc_no_copy(num_products *
                                       spec->num_components_real);
    this->d_data_int->realloc_no_copy(num_products * spec->num_components_int);
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
            this->d_data_real->ptr + (sym_offset + cx) * num_products;
        es.push(
            this->sycl_target->queue.fill(column_start, value, num_products));
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
            this->d_data_int->ptr + (sym_offset + cx) * num_products;
        es.push(
            this->sycl_target->queue.fill(column_start, value, num_products));
      }
    }

    es.wait();
  }
};

/**
 * Helper function to create a ProductMatrix shared pointer.
 *
 * TODO
 */
inline std::shared_ptr<ProductMatrix>
product_matrix(SYCLTargetSharedPtr sycl_target,
               std::shared_ptr<ProductMatrixSpec> spec) {
  return std::make_shared<ProductMatrix>(sycl_target, spec);
}

namespace ParticleLoopImplementation {
/**
 * Method to compute access to a ProductMatrix (read)
 */
inline ProductMatrixGetConst
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<ProductMatrix *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a ProductMatrix (write)
 */
inline ProductMatrixGet create_loop_arg(ParticleLoopGlobalInfo *global_info,
                                        sycl::handler &cgh,
                                        Access::Write<ProductMatrix *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a ProductMatrix (add)
 */
inline ProductMatrixGet create_loop_arg(ParticleLoopGlobalInfo *global_info,
                                        sycl::handler &cgh,
                                        Access::Add<ProductMatrix *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
