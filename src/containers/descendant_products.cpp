#include <neso_particles/containers/descendant_products.hpp>

namespace NESO::Particles {
DescendantProductsGet DescendantProducts::impl_get() {
  DescendantProductsGet dpg;
  dpg.product_matrix_get = ProductMatrix::impl_get();
  dpg.num_parent_particles = this->num_parent_particles;
  dpg.d_parent_cells = this->d_parent_cells->ptr;
  dpg.d_parent_layers = this->d_parent_layers->ptr;
  return dpg;
}

DescendantProducts::DescendantProducts(SYCLTargetSharedPtr sycl_target,
                                       std::shared_ptr<ProductMatrixSpec> spec,
                                       const int num_products_per_parent)
    : ProductMatrix(sycl_target, spec),
      num_products_per_parent(num_products_per_parent) {
  this->d_parent_cells = std::make_shared<BufferDevice<INT>>(sycl_target, 1);
  this->d_parent_layers = std::make_shared<BufferDevice<INT>>(sycl_target, 1);
}

void DescendantProducts::reset(const int num_parent_particles) {
  NESOASSERT(num_parent_particles >= 0,
             "A negative number of particles does not make sense.");
  const int num_products = num_parent_particles * num_products_per_parent;
  this->num_parent_particles = num_parent_particles;
  ProductMatrix::reset(num_products);
  if (num_parent_particles > 0) {
    this->d_parent_cells->realloc_no_copy(num_products);
    this->d_parent_layers->realloc_no_copy(num_products);
    auto e0 = this->sycl_target->queue.fill(this->d_parent_cells->ptr,
                                            static_cast<INT>(-1), num_products);
    auto e1 = this->sycl_target->queue.fill(this->d_parent_layers->ptr,
                                            static_cast<INT>(-1), num_products);
    e0.wait_and_throw();
    e1.wait_and_throw();
  }
}

} // namespace NESO::Particles
