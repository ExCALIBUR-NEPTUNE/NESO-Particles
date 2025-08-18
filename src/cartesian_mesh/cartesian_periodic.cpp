#include <neso_particles/cartesian_mesh/cartesian_periodic.hpp>
#include <neso_particles/common_impl.hpp>

namespace NESO::Particles {

CartesianPeriodic::CartesianPeriodic(SYCLTargetSharedPtr sycl_target,
                                     std::shared_ptr<CartesianHMesh> mesh,
                                     ParticleDatSharedPtr<REAL> position_dat)
    : d_extents(sycl_target, 3), sycl_target(sycl_target), mesh(mesh),
      position_dat(position_dat) {

  NESOASSERT(mesh->ndim <= 3, "bad mesh ndim");
  BufferHost<REAL> h_extents(sycl_target, 3);
  for (int dimx = 0; dimx < mesh->ndim; dimx++) {
    h_extents.ptr[dimx] = this->mesh->global_extents[dimx];
  }
  sycl_target->queue
      .memcpy(this->d_extents.ptr, h_extents.ptr, mesh->ndim * sizeof(REAL))
      .wait_and_throw();

  const int k_ndim = this->mesh->ndim;
  NESOASSERT(((k_ndim > 0) && (k_ndim < 4)), "Bad number of dimensions");
  const REAL *RESTRICT k_extents = this->d_extents.ptr;

  this->pbc_loop = particle_loop(
      "CartesianPeriodicPBC", position_dat,
      [=](auto P) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          const REAL pos = P[dimx];
          // offset the position in the current dimension to be
          // positive by adding a value times the extent
          const int n_extent_offset_int = abs((int)pos);
          const REAL tmp_extent = k_extents[dimx];
          const REAL n_extent_offset_real = n_extent_offset_int + 2;
          const REAL pos_fmod =
              sycl::fmod(pos + n_extent_offset_real * tmp_extent, tmp_extent);
          P[dimx] = pos_fmod;
        }
      },
      Access::write(position_dat));
}

CartesianPeriodic::CartesianPeriodic(std::shared_ptr<CartesianHMesh> mesh,
                                     ParticleGroupSharedPtr particle_group)
    : CartesianPeriodic(particle_group->sycl_target, mesh,
                        particle_group->position_dat) {}

void CartesianPeriodic::execute() { this->pbc_loop->execute(); }

} // namespace NESO::Particles
