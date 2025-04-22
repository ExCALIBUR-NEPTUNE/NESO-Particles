#ifndef _NESO_PARTICLES_PRODUCT_MATRIX_H_
#define _NESO_PARTICLES_PRODUCT_MATRIX_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../particle_spec.hpp"
#include "particle_set_device.hpp"
#include <map>
#include <memory>
#include <numeric>
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
  /**
   * Access a REAL product property.
   *
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering REAL
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @returns Constant reference to product value.
   */
  const REAL &at_real(const int product, const int property,
                      const int component) const {
    return ptr_real[(offsets_real[property] + component) * num_products +
                    product];
  }
  /**
   * Access a INT product property.
   *
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering INT
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @returns Constant reference to product value.
   */
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

  /**
   * Atomically increment a REAL product property.
   *
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering REAL
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @param value Value to increment property value by.
   * @returns Value prior to increment.
   */
  inline REAL fetch_add_real(const int product, const int property,
                             const int component, const REAL value) {
    return atomic_fetch_add(
        &ptr_real[(offsets_real[property] + component) * num_products +
                  product],
        value);
  }

  /**
   * Atomically increment a INT product property.
   *
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering INT
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @param value Value to increment property value by.
   * @returns Value prior to increment.
   */
  inline INT fetch_add_int(const int product, const int property,
                           const int component, const INT value) {
    return atomic_fetch_add(
        &ptr_int[(offsets_int[property] + component) * num_products + product],
        value);
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

  /**
   * Write access to REAL particle property.
   *
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering REAL
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @returns Modifiable reference to property value.
   */
  REAL &at_real(const int product, const int property, const int component) {
    return ptr_real[(offsets_real[property] + component) * num_products +
                    product];
  }

  /**
   * Write access to INT particle property.
   *
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering INT
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @returns Modifiable reference to property value.
   */
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
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
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
inline void
create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
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
typedef ParticleSetDeviceSpec ProductMatrixSpec;

/**
 * Helper function to create ProductMatrixSpec instances.
 *
 * @param particle_spec Specification for product particle properties.
 */
inline std::shared_ptr<ProductMatrixSpec>
product_matrix_spec(ParticleSpec particle_spec) {
  return std::make_shared<ProductMatrixSpec>(particle_spec);
}

/**
 * Type to store N products which can be passed to a ParticleLoop.
 * Fundamentally this class allocates two matrices, one for REAL valued
 * properties and one for INT value properties. These matrices are allocated
 * column major. Each output particle populates a row in these two matrices.
 * The column ordering is based on the ordering of properties and there
 * components in the input particle specification.
 */
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
  /// The specification of the products.
  std::shared_ptr<ProductMatrixSpec> spec;
  virtual ~ProductMatrix() = default;

  ProductMatrix() = default;

  /**
   * Note that the copy operator creates shallow copies of the array.
   */
  ProductMatrix &operator=(const ProductMatrix &) = default;

  /**
   * Create a new product matrix on the SYCLTarget. The reset method should be
   * called with the desired number of output particles before a loop is
   * executed which requires space for those output particles.
   *
   * @param sycl_target Device on which particle loops will be executed using
   * the product matrix.
   * @param spec A specification for the output particle properties.
   */
  ProductMatrix(SYCLTargetSharedPtr sycl_target,
                std::shared_ptr<ProductMatrixSpec> spec)
      : sycl_target(sycl_target), num_products(0), spec(spec) {

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
   * Allocate space for a number of particle properties and fill the matrix
   * with the default values.
   *
   * @param reset Number of output particles to set in matrix.
   */
  virtual inline void reset(const int num_products) {
    const auto spec = this->spec.get();
    NESOASSERT(spec != nullptr, "ProductMatrix is not initialised.");
    NESOASSERT(num_products >= 0,
               "A negative number of products does not make sense.");
    this->num_products = num_products;
    if (num_products > 0) {
      this->d_data_real->realloc_no_copy(num_products *
                                         spec->num_components_real);
      this->d_data_int->realloc_no_copy(num_products *
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
  }
};

/**
 * Helper function to create ProductMatrix shared pointer.
 *
 * @param sycl_target Device on which particle loops will be executed using
 * the product matrix.
 * @param spec A specification for the output particle properties.
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
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Read<ProductMatrix *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a ProductMatrix (write)
 */
inline ProductMatrixGet
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Write<ProductMatrix *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a ProductMatrix (add)
 */
inline ProductMatrixGet
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,
                [[maybe_unused]] sycl::handler &cgh,
                Access::Add<ProductMatrix *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
