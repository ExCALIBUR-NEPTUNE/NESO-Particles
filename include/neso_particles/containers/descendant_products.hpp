#ifndef _NESO_PARTICLES_DESCENDANT_PRODUCTS_H_
#define _NESO_PARTICLES_DESCENDANT_PRODUCTS_H_

#include "../compute_target.hpp"
#include "../loop/access_descriptors.hpp"
#include "../loop/particle_loop_base.hpp"
#include "../loop/particle_loop_index.hpp"
#include "../particle_spec.hpp"
#include "product_matrix.hpp"

namespace NESO::Particles {

class DescendantProducts;

struct DescendantProductsGet {
  ProductMatrixGet product_matrix_get;
  int num_particles;
  INT *d_parent_cells;
  INT *d_parent_layers;
};

/**
 *  Defines the access implementations and types for DescendantProducts objects.
 */
namespace Access::DescendantProducts {
/**
 * ParticleLoop access type for DescendantProducts Write access.
 */
struct Write {
  /// Pointer to underlying data for the array.
  Write() = default;
  REAL *ptr_real;
  INT *ptr_int;
  int const *offsets_real;
  int const *offsets_int;
  int num_products;
  int num_particles;
  INT *d_parent_cells;
  INT *d_parent_layers;

  /**
   * Write access to REAL particle property.
   *
   * @param particle_index The index of the particle creating this product.
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering REAL
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @returns Modifiable reference to property value.
   */
  REAL &at_real(const LoopIndex::Read &particle_index, const int product,
                const int property, const int component) {
    const int sub_index = particle_index.get_loop_linear_index();
    const int matrix_row = sub_index + product * num_particles;
    return ptr_real[(offsets_real[property] + component) * num_products +
                    matrix_row];
  }

  /**
   * Write access to INT particle property.
   *
   * @param particle_index The index of the particle creating this product.
   * @param product The index of the product to access, i.e. the row in the
   * product matrix.
   * @param property The particle property to access using the ordering INT
   * ParticleProp instances were passed to the ProductMatrixSpec
   * @param component The component of the property to access.
   * @returns Modifiable reference to property value.
   */
  INT &at_int(const LoopIndex::Read &particle_index, const int product,
              const int property, const int component) {
    const int sub_index = particle_index.get_loop_linear_index();
    const int matrix_row = sub_index + product * num_particles;
    return ptr_int[(offsets_int[property] + component) * num_products +
                   matrix_row];
  }

  /**
   * Set the parent of a product.
   *
   * @param particle_index Index of parent particle.
   * @param product Index of child property to set as descendant of parent.
   */
  inline void set_parent(const LoopIndex::Read &particle_index,
                         const int product) {
    const int sub_index = particle_index.get_loop_linear_index();
    const int matrix_row = sub_index + product * num_particles;
    this->d_parent_cells[matrix_row] = particle_index.cell;
    this->d_parent_layers[matrix_row] = particle_index.layer;
  }
};

} // namespace Access::DescendantProducts

namespace ParticleLoopImplementation {
/**
 *  Loop parameter for write access of a DescendantProducts.
 */
template <> struct LoopParameter<Access::Write<DescendantProducts>> {
  using type = DescendantProductsGet;
};
/**
 *  KernelParameter type for write access to a DescendantProducts.
 */
template <> struct KernelParameter<Access::Write<DescendantProducts>> {
  using type = Access::DescendantProducts::Write;
};
/**
 *  Function to create the kernel argument for DescendantProducts write access.
 */
inline void create_kernel_arg([[maybe_unused]] ParticleLoopIteration &iterationx,
                              DescendantProductsGet &rhs,
                              Access::DescendantProducts::Write &lhs) {
  lhs.ptr_real = rhs.product_matrix_get.ptr_real;
  lhs.ptr_int = rhs.product_matrix_get.ptr_int;
  lhs.offsets_real = rhs.product_matrix_get.offsets_real;
  lhs.offsets_int = rhs.product_matrix_get.offsets_int;
  lhs.num_products = rhs.product_matrix_get.num_products;
  lhs.num_particles = rhs.num_particles;
  lhs.d_parent_cells = rhs.d_parent_cells;
  lhs.d_parent_layers = rhs.d_parent_layers;
}

/**
 * Method to compute access to a DescendantProducts (write)
 */
inline DescendantProductsGet
create_loop_arg(ParticleLoopGlobalInfo *global_info, sycl::handler &cgh,
                Access::Write<DescendantProducts *> &a);
} // namespace ParticleLoopImplementation

/**
 * Class to create M products from each parent particle. Products may inherit
 * property values from the parent.
 */
class DescendantProducts : public ProductMatrix {
  friend class ParticleGroup;
  friend inline DescendantProductsGet
  ParticleLoopImplementation::create_loop_arg(
      ParticleLoopImplementation::ParticleLoopGlobalInfo *global_info,
      sycl::handler &cgh, Access::Write<DescendantProducts *> &a);

protected:
  std::shared_ptr<BufferDevice<INT>> d_parent_cells;
  std::shared_ptr<BufferDevice<INT>> d_parent_layers;

  inline DescendantProductsGet impl_get() {
    DescendantProductsGet dpg;
    dpg.product_matrix_get = ProductMatrix::impl_get();
    dpg.num_particles = this->num_particles;
    dpg.d_parent_cells = this->d_parent_cells->ptr;
    dpg.d_parent_layers = this->d_parent_layers->ptr;
    return dpg;
  }

public:
  DescendantProducts() = default;
  DescendantProducts &operator=(const DescendantProducts &) = default;

  /// The number of products this data structure is expecting per parent
  /// particle.
  int num_products_per_parent;
  /// The current number of parent particles the container is allocated for.
  int num_particles;

  /**
   * Create an instance from a ProductMatrixSpec that defines which particle
   * properties should be available for writing in the particle loop kernel.
   *
   * @param sycl_target Compute device on which products will be made.
   * @param spec Specification of properties which require specialisation in
   * the creating kernel.
   * @param num_products_per_parent The maximum number of products required per
   * parent particle.
   */
  DescendantProducts(SYCLTargetSharedPtr sycl_target,
                     std::shared_ptr<ProductMatrixSpec> spec,
                     const int num_products_per_parent)
      : ProductMatrix(sycl_target, spec),
        num_products_per_parent(num_products_per_parent), num_particles(0) {
    this->d_parent_cells = std::make_shared<BufferDevice<INT>>(sycl_target, 1);
    this->d_parent_layers = std::make_shared<BufferDevice<INT>>(sycl_target, 1);
  }

  /**
   * Resize the container to hold space for the products of a number of parent
   * particles.
   *
   * @param num_particles Number of parent particles space is required for.
   */
  virtual inline void reset(const int num_particles) override {
    this->num_particles = num_particles;
    const int num_products = num_particles * num_products_per_parent;
    ProductMatrix::reset(num_products);
    this->d_parent_cells->realloc_no_copy(num_products);
    this->d_parent_layers->realloc_no_copy(num_products);
    auto e0 = this->sycl_target->queue.fill(this->d_parent_cells->ptr,
                                            static_cast<INT>(-1), num_products);
    auto e1 = this->sycl_target->queue.fill(this->d_parent_layers->ptr,
                                            static_cast<INT>(-1), num_products);
    e0.wait_and_throw();
    e1.wait_and_throw();
  };
};

namespace ParticleLoopImplementation {

/**
 * Method to compute access to a DescendantProducts (write)
 */
inline DescendantProductsGet
create_loop_arg([[maybe_unused]] ParticleLoopGlobalInfo *global_info,[[maybe_unused]]  sycl::handler &cgh,
                Access::Write<DescendantProducts *> &a) {
  return a.obj->impl_get();
}

} // namespace ParticleLoopImplementation

} // namespace NESO::Particles

#endif
