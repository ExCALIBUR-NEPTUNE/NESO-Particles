#include <neso_particles/algorithms/subdivide_cartesian_cells.hpp>

namespace NESO::Particles {

SubdivideCartesianCells::SubdivideCartesianCells(
    SYCLTargetSharedPtr sycl_target, CartesianHMeshSharedPtr mesh,
    std::vector<int> &num_subdivisions)
    : sycl_target(sycl_target), mesh(mesh), num_subdivisions(num_subdivisions) {

  NESOASSERT(mesh != nullptr, "Bad mesh pointer passed.");

  const int cell_count = mesh->get_cell_count();
  const int ndim = mesh->get_ndim();

  NESOASSERT(cell_count == num_subdivisions.size(),
             "Miss-match in cell count between mesh and vector providing "
             "number of subdivisions.");

  this->d_num_subdivisions = std::make_shared<BufferDevice<int>>(
      this->sycl_target, this->num_subdivisions);
  const int *k_num_subdivisions = d_num_subdivisions->ptr;

  this->d_subdvision_inverse_widths =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, cell_count);
  REAL *k_subdvision_inverse_widths = d_subdvision_inverse_widths->ptr;

  const REAL cell_width = this->mesh->cell_width_fine;

  auto e0 = this->sycl_target->queue.parallel_for(
      this->sycl_target->device_limits.validate_range_global(
          sycl::range<1>(cell_count)),
      [=](auto ix) {
        k_subdvision_inverse_widths[ix] = k_num_subdivisions[ix] / cell_width;
      });

  auto h_owned_cells = this->mesh->get_owned_cells();
  std::vector<REAL> h_origins(cell_count * ndim);
  for (int dx = 0; dx < ndim; dx++) {
    for (int cellx = 0; cellx < cell_count; cellx++) {
      const auto index = h_owned_cells.at(cellx).at(dx);
      h_origins.at(dx * cell_count + cellx) = cell_width * index;
    }
  }

  this->d_origins =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, h_origins);
  e0.wait_and_throw();
}

template void
SubdivideCartesianCells::map(std::shared_ptr<ParticleGroup> particle_sub_group,
                             Sym<INT> sym_name, const int sym_component);

template void SubdivideCartesianCells::map(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);

} // namespace NESO::Particles
