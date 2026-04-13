#include <neso_particles/algorithms/subdivide_cartesian_cells.hpp>

namespace NESO::Particles {

SubdivideCartesianCells::SubdivideCartesianCells(
    SYCLTargetSharedPtr sycl_target, CartesianHMeshSharedPtr mesh,
    std::vector<int> &sub_cell_count)
    : sycl_target(sycl_target), mesh(mesh), sub_cell_count(sub_cell_count) {

  NESOASSERT(mesh != nullptr, "Bad mesh pointer passed.");

  const int cell_count = mesh->get_cell_count();
  const int ndim = mesh->get_ndim();

  NESOASSERT(cell_count == sub_cell_count.size(),
             "Miss-match in cell count between mesh and vector providing "
             "number of subdivisions.");

  this->d_sub_cell_count = std::make_shared<BufferDevice<int>>(
      this->sycl_target, this->sub_cell_count);
  const int *k_sub_cell_count = d_sub_cell_count->ptr;

  this->d_sub_cell_inverse_widths =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, cell_count);
  REAL *k_subdvision_inverse_widths = d_sub_cell_inverse_widths->ptr;

  const REAL cell_width = this->mesh->cell_width_fine;

  auto e0 = this->sycl_target->queue.parallel_for(
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<1>(cell_count)),
      [=](auto ix) {
        k_subdvision_inverse_widths[ix] = k_sub_cell_count[ix] / cell_width;
      });

  auto h_owned_cells = this->mesh->get_owned_cells();
  std::vector<REAL> h_origins(cell_count * ndim);
  this->h_num_subdivision_cells.resize(cell_count);
  for (int cellx = 0; cellx < cell_count; cellx++) {
    NESOASSERT(this->sub_cell_count.at(cellx) > 0,
               "A non-positive sub cell count does not make sense.");
    this->h_num_subdivision_cells.at(cellx) =
        std::pow(this->sub_cell_count.at(cellx), ndim);
    for (int dx = 0; dx < ndim; dx++) {
      const auto index = h_owned_cells.at(cellx).at(dx);
      h_origins.at(cellx * ndim + dx) = cell_width * index;
    }
  }

  this->d_origins =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, h_origins);
  e0.wait_and_throw();
}

const std::vector<int> &SubdivideCartesianCells::get_num_subdivision_cells() {
  return this->h_num_subdivision_cells;
}

template void
SubdivideCartesianCells::map(std::shared_ptr<ParticleGroup> particle_sub_group,
                             Sym<INT> sym_name, const int sym_component);

template void SubdivideCartesianCells::map(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);

} // namespace NESO::Particles
