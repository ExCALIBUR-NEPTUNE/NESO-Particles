#include <neso_particles/cartesian_mesh/subdivide_cells_cartesian_h_mesh.hpp>

namespace NESO::Particles {

SubdivideCellsCartesianHMesh::~SubdivideCellsCartesianHMesh() {
  this->event_stack.wait();
}

SubdivideCellsCartesianHMesh::SubdivideCellsCartesianHMesh(
    SYCLTargetSharedPtr sycl_target, CartesianHMeshSharedPtr mesh)
    : sycl_target(sycl_target), mesh(mesh),
      h_num_subdivisions(std::vector<int, HostAllocator<int>>(
          mesh->get_cell_count(), HostAllocator<int>(sycl_target->queue))),
      h_num_collision_cells(std::vector<int>(mesh->get_cell_count())) {

  const auto num_cells = mesh->get_cell_count();
  const int k_ndim = this->mesh->get_ndim();
  NESOASSERT((0 < k_ndim) && (k_ndim < 4), "Unexpected number of dimensions.");

  this->d_num_subdivisions =
      std::make_shared<BufferDevice<int>>(this->sycl_target, num_cells);
  this->d_num_collision_cells =
      std::make_shared<BufferDevice<int>>(this->sycl_target, num_cells);
  this->d_origins = std::make_shared<BufferDevice<REAL>>(this->sycl_target,
                                                         num_cells * k_ndim);
  this->d_inverse_cell_widths =
      std::make_shared<BufferDevice<REAL>>(this->sycl_target, num_cells);

  auto cells = this->mesh->get_owned_cells();

  std::vector<REAL, HostAllocator<REAL>> h_origins(
      num_cells * k_ndim, HostAllocator<REAL>(this->sycl_target->queue));

  const REAL cell_width = this->mesh->get_cell_width_fine();
  for (int cellx = 0; cellx < num_cells; cellx++) {
    for (int dx = 0; dx < k_ndim; dx++) {
      h_origins.at(dx * num_cells + cellx) =
          cells.at(cellx).at(dx) * cell_width;
    }
  }

  this->sycl_target->queue
      .memcpy(this->d_origins->ptr, h_origins.data(),
              num_cells * k_ndim * sizeof(REAL))
      .wait_and_throw();
}

void SubdivideCellsCartesianHMesh::set_num_subdivisions(
    const std::vector<int> &num_subdivisions) {

  this->event_stack.wait();

  const auto num_cells = this->mesh->get_cell_count();
  const int k_ndim = this->mesh->get_ndim();

  NESOASSERT(num_subdivisions.size() >= num_cells,
             "Insufficiently sized vector passed.");
  std::memcpy(this->h_num_subdivisions.data(), num_subdivisions.data(),
              sizeof(int) * num_cells);

  int *k_num_subdivisions = this->d_num_subdivisions->ptr;
  int *k_num_collision_cells = this->d_num_collision_cells->ptr;
  REAL *k_inverse_cell_widths = this->d_inverse_cell_widths->ptr;

  auto e0 = this->sycl_target->queue.memcpy(k_num_subdivisions,
                                            this->h_num_subdivisions.data(),
                                            sizeof(int) * num_cells);

  const REAL k_cell_width_fine = this->mesh->get_cell_width_fine();

  auto e1 = this->sycl_target->queue.parallel_for(
      sycl::range<1>(num_cells), e0, [=](sycl::item<1> idx) {
        const int s = k_num_subdivisions[idx];
        const int d = 1 << s;
        int dd[3] = {d, d * d, d * d * d};
        k_num_collision_cells[idx] = dd[k_ndim - 1];
        k_inverse_cell_widths[idx] = static_cast<REAL>(d) / k_cell_width_fine;
      });

  auto e2 = this->sycl_target->queue.memcpy(this->h_num_collision_cells.data(),
                                            k_num_collision_cells,
                                            sizeof(int) * num_cells, e1);
  this->event_stack.push(e2);
}

template void SubdivideCellsCartesianHMesh::map(
    std::shared_ptr<ParticleGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);

template void SubdivideCellsCartesianHMesh::map(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);

const std::vector<int> &
SubdivideCellsCartesianHMesh::get_num_subdivision_cells() {
  this->event_stack.wait();
  return this->h_num_collision_cells;
}
} // namespace NESO::Particles
