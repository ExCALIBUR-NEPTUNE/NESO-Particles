#ifndef _NESO_PARTICLES_PRODUCT_MATRIX_H_
#define _NESO_PARTICLES_PRODUCT_MATRIX_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "local_array.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace NESO::Particles {

template <typename T> class ProductMatrix;

template <typename T> struct ProductMatrixGet {
  T *ptr;
  std::size_t num_products;
};

template <typename T> struct ProductMatrixGetConst {
  T const *ptr;
  std::size_t num_products;
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
template <typename T> struct Read {
  /// Pointer to underlying data for the array.
  Read() = default;
  T const *ptr;
  std::size_t num_products;
  template <typename U, typename V>
  const T at(const U product, const V component) const {
    return ptr[component * num_products + product];
  }
};

/**
 * ParticleLoop access type for ProductMatrix Add access.
 */
template <typename T> struct Add {
  /// Pointer to underlying data for the array.
  Add() = default;
  T *ptr;
  std::size_t num_products;
  /**
   * The local array is local to the MPI rank where the partial sum is a
   * meaningful value.
   */
  template <typename U, typename V>
  inline T fetch_add(const U product, const V component, const T value) const {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>
        element_atomic(ptr[component * num_products + product]);
    return element_atomic.fetch_add(value);
  }
};

/**
 * ParticleLoop access type for ProductMatrix Write access.
 */
template <typename T> struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  T *ptr;
  std::size_t num_products;
  template <typename U, typename V>
  T &at(const U product, const V component) const {
    return ptr[component * num_products + product];
  }
};

} // namespace Access::ProductMatrix

namespace ParticleLoopImplementation {

/**
 *  Loop parameter for read access of a ProductMatrix.
 */
template <typename T> struct LoopParameter<Access::Read<ProductMatrix<T>>> {
  using type = ProductMatrixGetConst<T>;
};
/**
 *  Loop parameter for write access of a ProductMatrix.
 */
template <typename T> struct LoopParameter<Access::Write<ProductMatrix<T>>> {
  using type = ProductMatrixGet<T>;
};
/**
 *  Loop parameter for add access of a ProductMatrix.
 */
template <typename T> struct LoopParameter<Access::Add<ProductMatrix<T>>> {
  using type = ProductMatrixGet<T>;
};
/**
 *  KernelParameter type for read access to a ProductMatrix.
 */
template <typename T> struct KernelParameter<Access::Read<ProductMatrix<T>>> {
  using type = Access::ProductMatrix::Read<T>;
};
/**
 *  KernelParameter type for write access to a ProductMatrix.
 */
template <typename T> struct KernelParameter<Access::Write<ProductMatrix<T>>> {
  using type = Access::ProductMatrix::Write<T>;
};
/**
 *  KernelParameter type for add access to a ProductMatrix.
 */
template <typename T> struct KernelParameter<Access::Add<ProductMatrix<T>>> {
  using type = Access::ProductMatrix::Add<T>;
};
/**
 *  Function to create the kernel argument for ProductMatrix read access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              ProductMatrixGetConst<T> &rhs,
                              Access::ProductMatrix::Read<T> &lhs) {
  lhs.num_products = rhs.num_products;
  lhs.ptr = rhs.ptr;
}
/**
 *  Function to create the kernel argument for ProductMatrix write access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              ProductMatrixGet<T> &rhs,
                              Access::ProductMatrix::Write<T> &lhs) {
  lhs.num_products = rhs.num_products;
  lhs.ptr = rhs.ptr;
}
/**
 *  Function to create the kernel argument for ProductMatrix add access.
 */
template <typename T>
inline void create_kernel_arg(ParticleLoopIteration &iterationx,
                              ProductMatrixGet<T> &rhs,
                              Access::ProductMatrix::Add<T> &lhs) {
  lhs.num_products = rhs.num_products;
  lhs.ptr = rhs.ptr;
}

/**
 * Method to compute access to a ProductMatrix (read)
 */
template <typename T>
inline ProductMatrixGetConst<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Read<ProductMatrix<T> *> &a) {
  return a.obj->impl_get_const();
}
/**
 * Method to compute access to a ProductMatrix (write)
 */
template <typename T>
inline ProductMatrixGet<T>
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<ProductMatrix<T> *> &a) {
  return a.obj->impl_get();
}
/**
 * Method to compute access to a ProductMatrix (add)
 */
template <typename T>
inline ProductMatrixGet<T> create_loop_arg(ParticleLoopGlobalInfo *global_info,
                                           sycl::handler &cgh,
                                           Access::Add<ProductMatrix<T> *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

template <typename T> class ProductMatrix {

  friend ProductMatrixGetConst<T>
  ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Read<ProductMatrix<T> *> &a);
  friend ProductMatrixGet<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<ProductMatrix<T> *> &a);
  friend ProductMatrixGet<T> ParticleLoopImplementation::create_loop_arg<T>(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Add<ProductMatrix<T> *> &a);

protected:
  std::shared_ptr<BufferDevice<T>> d_data;

  inline ProductMatrixGet<T> impl_get() {
    return {this->d_data->ptr, this->num_products};
  }
  inline ProductMatrixGetConst<T> impl_get_const() {
    return {this->d_data->ptr, this->num_products};
  }

public:
  /// The SYCLTarget products are created on.
  SYCLTargetSharedPtr sycl_target;
  /// The number of products stored.
  int num_products;
  /// The number of components of type T stored.
  int num_components;

  ProductMatrix() = default;

  /**
   * Note that the copy operator creates shallow copies of the array.
   */
  ProductMatrix<T> &operator=(const ProductMatrix<T> &) = default;

  /**
   * TODO
   */
  ProductMatrix(SYCLTargetSharedPtr sycl_target, const std::size_t num_products,
                const std::size_t num_components)
      : sycl_target(sycl_target), num_products(num_products),
        num_components(num_components),
        d_data(std::make_shared<BufferDevice<T>>(
            sycl_target, num_products * num_components)) {}

  /**
   * TODO
   */
  inline void realloc_no_copy(const std::size_t num_products,
                              const std::size_t num_components) {
    this->num_products = num_products;
    this->num_components = num_components;
    this->d_data->realloc_no_copy(num_products * num_components);
  }
};
} // namespace NESO::Particles

#endif
