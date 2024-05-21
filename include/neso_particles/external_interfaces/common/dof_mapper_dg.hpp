#ifndef _NESO_PARTICLES_DOF_MAPPER_DG_H_
#define _NESO_PARTICLES_DOF_MAPPER_DG_H_
#include "../../compute_target.hpp"
#include "../../containers/cell_dat_const.hpp"
#include "../../typedefs.hpp"
#include <vector>

namespace NESO::Particles::ExternalCommon {

/**
 * TODO
 */
struct DOFMapperDGDeviceMapper {
  int num_cells_local;
  int num_dofs_per_cell;
  int *d_map_to_index;

  /**
   * TODO
   */
  inline int get(const int cell, const int dof) const {
    return this->d_map_to_index[cell * num_dofs_per_cell + dof];
  }

  /**
   * TODO
   */
  inline void copy_to_external(const int cell, const int dof,
                               const REAL *RESTRICT source,
                               REAL *RESTRICT destination) const {
    const int index_source = cell * num_dofs_per_cell + dof;
    const int index_dst = this->get(cell, dof);
    destination[index_dst] = source[index_source];
  }

  /**
   * TODO
   */
  inline void copy_from_external(const int cell, const int dof,
                                 const REAL *RESTRICT source,
                                 REAL *RESTRICT destination) const {
    const int index_dst = cell * num_dofs_per_cell + dof;
    const int index_source = this->get(cell, dof);
    destination[index_dst] = source[index_source];
  }
};

/**
 * TODO
 */
class DOFMapperDG {
protected:
  std::vector<int> map_to_index;
  std::unique_ptr<BufferDevice<int>> d_map_to_index;
  std::unique_ptr<BufferDevice<REAL>> d_dofs;
  bool device_valid;

public:
  SYCLTargetSharedPtr sycl_target;
  int num_cells_local;
  int num_dofs_per_cell;

  DOFMapperDG() = default;

  /**
   * TODO
   */
  DOFMapperDG(SYCLTargetSharedPtr sycl_target, const int num_cells_local,
              const int num_dofs_per_cell)
      : sycl_target(sycl_target), num_cells_local(num_cells_local),
        num_dofs_per_cell(num_dofs_per_cell) {
    this->map_to_index.resize(num_cells_local * num_dofs_per_cell);
    std::fill(this->map_to_index.begin(), this->map_to_index.end(), -1);
    this->device_valid = false;
  }

  /**
   * TODO
   */
  inline int index(const int cell, const int dof) const {
    NESOASSERT((0 <= cell) && (cell < this->num_cells_local),
               "Bad cell passed: " + std::to_string(cell));
    NESOASSERT((0 <= dof) && (dof < this->num_dofs_per_cell),
               "Bad dof passed: " + std::to_string(cell));
    return cell * this->num_dofs_per_cell + dof;
  }

  /**
   * TODO
   */
  inline void set(const int cell, const int dof, const int index) {
    this->map_to_index.at(this->index(cell, dof)) = index;
    this->device_valid = false;
  }

  /**
   * TODO
   */
  inline int get(const int cell, const int dof) const {
    return this->map_to_index.at(this->index(cell, dof));
  }

  /**
   * TODO
   */
  inline DOFMapperDGDeviceMapper get_device_mapper() {
    if (!this->device_valid) {
      this->d_map_to_index = std::make_unique<BufferDevice<int>>(
          this->sycl_target, this->map_to_index);
    }

    DOFMapperDGDeviceMapper m;
    m.num_cells_local = this->num_cells_local;
    m.num_dofs_per_cell = this->num_dofs_per_cell;
    m.d_map_to_index = this->d_map_to_index->ptr;
    return m;
  }

  /**
   * TODO
   */
  inline void copy_to_external(CellDatConstSharedPtr<REAL> cell_dat_const,
                               REAL *h_external_dofs, EventStack &es) {
    const auto k_mapper = this->get_device_mapper();
    const int k_num_cells_local = this->num_cells_local;
    const int k_num_dofs_per_cell = this->num_dofs_per_cell;
    if (!this->d_dofs) {
      this->d_dofs = std::make_unique<BufferDevice<REAL>>(
          this->sycl_target, k_num_cells_local * k_num_dofs_per_cell);
    }
    auto k_dofs = this->d_dofs->ptr;
    auto k_cell_dat_const = cell_dat_const->device_ptr();

    auto e0 = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<2>(k_num_cells_local, k_num_dofs_per_cell),
                         [=](sycl::id<2> idx) {
                           const int cell = idx[0];
                           const int dof = idx[1];
                           k_mapper.copy_to_external(cell, dof,
                                                     k_cell_dat_const, k_dofs);
                         });
    });

    const std::size_t num_bytes =
        k_num_cells_local * k_num_dofs_per_cell * sizeof(REAL);
    auto e1 =
        this->sycl_target->queue.memcpy(h_external_dofs, k_dofs, num_bytes, e0);
    es.push(e1);
  }

  /**
   * TODO
   */
  inline void copy_from_external(CellDatConstSharedPtr<REAL> cell_dat_const,
                                 REAL *h_external_dofs, EventStack &es) {
    const auto k_mapper = this->get_device_mapper();
    const int k_num_cells_local = this->num_cells_local;
    const int k_num_dofs_per_cell = this->num_dofs_per_cell;
    if (!this->d_dofs) {
      this->d_dofs = std::make_unique<BufferDevice<REAL>>(
          this->sycl_target, k_num_cells_local * k_num_dofs_per_cell);
    }
    auto k_dofs = this->d_dofs->ptr;
    auto k_cell_dat_const = cell_dat_const->device_ptr();

    const std::size_t num_bytes =
        k_num_cells_local * k_num_dofs_per_cell * sizeof(REAL);
    this->sycl_target->queue.memcpy(h_external_dofs, k_dofs, num_bytes)
        .wait_and_throw();

    auto e1 = this->sycl_target->queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<>(sycl::range<2>(k_num_cells_local, k_num_dofs_per_cell),
                         [=](sycl::id<2> idx) {
                           const int cell = idx[0];
                           const int dof = idx[1];
                           k_mapper.copy_from_external(cell, dof, k_dofs,
                                                       k_cell_dat_const);
                         });
    });

    es.push(e1);
  }
};

} // namespace NESO::Particles::ExternalCommon

#endif
