#include <neso_particles/cartesian_mesh/cartesian_cell_bin.hpp>
#include <neso_particles/common_impl.hpp>
#include <neso_particles/loop/particle_loop_functions.hpp>

namespace NESO::Particles {

ParticleLoopSharedPtr
CartesianCellBin::get_loop(ParticleDatSharedPtr<REAL> position_dat,
                           ParticleDatSharedPtr<INT> cell_id_dat) {
  const int k_ndim = this->mesh->ndim;
  NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
  auto k_inverse_cell_width_fine = this->mesh->inverse_cell_width_fine;
  auto k_cell_width_fine = this->mesh->cell_width_fine;

  auto k_cell_counts = this->d_cell_counts.ptr;
  auto k_cell_starts = this->d_cell_starts.ptr;
  auto k_cell_ends = this->d_cell_ends.ptr;

  return particle_loop(
      "CartesianCellBin", position_dat,
      [=](auto positions, auto cell_id) {
        int cell_tmps[3] = {0, 0, 0};

        for (int dimx = 0; dimx < k_ndim; dimx++) {
          const REAL pos =
              positions[dimx] - k_cell_starts[dimx] * k_cell_width_fine;
          int cell_tmp = ((REAL)pos * k_inverse_cell_width_fine);
          cell_tmp = (cell_tmp < 0) ? 0 : cell_tmp;
          cell_tmp = (cell_tmp >= k_cell_ends[dimx]) ? k_cell_ends[dimx] - 1
                                                     : cell_tmp;
          cell_tmps[dimx] = cell_tmp;
        }

        // convert to linear index
        int linear_index = cell_tmps[k_ndim - 1];
        for (int dimx = k_ndim - 2; dimx >= 0; dimx--) {
          linear_index *= k_cell_counts[dimx];
          linear_index += cell_tmps[dimx];
        }
        cell_id[0] = linear_index;
      },
      Access::read(position_dat), Access::write(cell_id_dat));
}

CartesianCellBin::CartesianCellBin(SYCLTargetSharedPtr sycl_target,
                                   CartesianHMeshSharedPtr mesh)
    : d_cell_counts(sycl_target, 3), d_cell_starts(sycl_target, 3),
      d_cell_ends(sycl_target, 3), sycl_target(sycl_target), mesh(mesh),
      position_dat(nullptr), cell_id_dat(nullptr) {

  NESOASSERT(mesh->ndim <= 3, "bad mesh ndim");
  BufferHost<int> h_cell_counts(sycl_target, 3);
  for (int dimx = 0; dimx < mesh->ndim; dimx++) {
    h_cell_counts.ptr[dimx] = this->mesh->cell_counts_local[dimx];
  }
  sycl_target->queue
      .memcpy(this->d_cell_counts.ptr, h_cell_counts.ptr,
              mesh->ndim * sizeof(int))
      .wait_and_throw();
  sycl_target->queue
      .memcpy(this->d_cell_starts.ptr, mesh->cell_starts,
              mesh->ndim * sizeof(int))
      .wait_and_throw();
  sycl_target->queue
      .memcpy(this->d_cell_ends.ptr, mesh->cell_ends, mesh->ndim * sizeof(int))
      .wait_and_throw();
}

CartesianCellBin::CartesianCellBin(SYCLTargetSharedPtr sycl_target,
                                   CartesianHMeshSharedPtr mesh,
                                   ParticleDatSharedPtr<REAL> position_dat,
                                   ParticleDatSharedPtr<INT> cell_id_dat)
    : CartesianCellBin(sycl_target, mesh) {

  this->position_dat = position_dat;
  this->cell_id_dat = cell_id_dat;
}

void CartesianCellBin::execute() {
  NESOASSERT(this->cell_id_dat != nullptr,
             "Cell ID Dat not set. Maybe wrong constructor?");
  NESOASSERT(this->position_dat != nullptr,
             "Position Dat not set. Maybe wrong constructor?");
  auto r = ProfileRegion("CartesianCellBin", "execute");
  auto loop = this->get_loop(position_dat, cell_id_dat);
  loop->execute();
  r.end();
  this->sycl_target->profile_map.add_region(r);
}

void CartesianCellBin::map_cells(ParticleGroup &particle_group,
                                 [[maybe_unused]] const int map_cell) {
  this->get_loop(particle_group.position_dat, particle_group.cell_id_dat)
      ->execute();
}

} // namespace NESO::Particles
