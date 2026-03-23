#include <neso_particles/cartesian_mesh/cartesian_periodic.hpp>
#include <neso_particles/common_impl.hpp>
#include <neso_particles/containers/local_array.hpp>

namespace NESO::Particles {

CartesianPeriodic::CartesianPeriodic(SYCLTargetSharedPtr sycl_target,
                                     std::shared_ptr<CartesianHMesh> mesh,
                                     ParticleDatSharedPtr<REAL> position_dat) {

  const int k_ndim = mesh->ndim;
  std::vector<REAL> extents(k_ndim);
  for (int dimx = 0; dimx < mesh->ndim; dimx++) {
    extents[dimx] = mesh->global_extents[dimx];
  }
  auto la_extents = std::make_shared<LocalArray<REAL>>(sycl_target, extents);
  for (int dimx = 0; dimx < mesh->ndim; dimx++) {
    extents[dimx] = 1.0 / mesh->global_extents[dimx];
  }
  auto la_inverse_extents =
      std::make_shared<LocalArray<REAL>>(sycl_target, extents);

  this->pbc_loop = particle_loop(
      "CartesianPeriodicPBC", position_dat,
      [=](auto P, auto E, auto IE) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          const REAL pos = P[dimx];
          // offset the position in the current dimension to be
          // positive by adding a value times the extent
          const int n_extent_offset_int = Kernel::abs((int)(pos * IE.at(dimx)));
          const REAL tmp_extent = E.at(dimx);
          const REAL n_extent_offset_real = n_extent_offset_int + 2;
          const REAL pos_fmod =
              Kernel::fmod(pos + n_extent_offset_real * tmp_extent, tmp_extent);
          P[dimx] = pos_fmod;
        }
      },
      Access::write(position_dat), Access::read(la_extents),
      Access::read(la_inverse_extents));
}

CartesianPeriodic::CartesianPeriodic(SYCLTargetSharedPtr sycl_target,
                                     std::shared_ptr<CartesianHMesh> mesh,
                                     ParticleDatSharedPtr<REAL> position_dat,
                                     std::vector<int> masks) {

  const int k_ndim = mesh->ndim;
  NESOASSERT(static_cast<std::size_t>(k_ndim) <= masks.size(),
             "Masks vector is too small.");

  std::vector<REAL> extents(k_ndim);
  for (int dimx = 0; dimx < mesh->ndim; dimx++) {
    extents[dimx] = mesh->global_extents[dimx];
  }
  auto la_extents = std::make_shared<LocalArray<REAL>>(sycl_target, extents);

  for (int dimx = 0; dimx < mesh->ndim; dimx++) {
    extents[dimx] = 1.0 / mesh->global_extents[dimx];
  }
  auto la_inverse_extents =
      std::make_shared<LocalArray<REAL>>(sycl_target, extents);

  auto la_masks = std::make_shared<LocalArray<int>>(sycl_target, masks);

  this->pbc_loop = particle_loop(
      "CartesianPeriodicPBCMasks", position_dat,
      [=](auto P, auto M, auto E, auto IE) {
        for (int dimx = 0; dimx < k_ndim; dimx++) {
          if (M.at(dimx)) {
            const REAL pos = P[dimx];
            // offset the position in the current dimension to be
            // positive by adding a value times the extent
            const int n_extent_offset_int =
                Kernel::abs((int)(pos * IE.at(dimx)));
            const REAL tmp_extent = E.at(dimx);
            const REAL n_extent_offset_real = n_extent_offset_int + 2;
            const REAL pos_fmod = Kernel::fmod(
                pos + n_extent_offset_real * tmp_extent, tmp_extent);
            P[dimx] = pos_fmod;
          }
        }
      },
      Access::write(position_dat), Access::read(la_masks),
      Access::read(la_extents), Access::read(la_inverse_extents));
}

CartesianPeriodic::CartesianPeriodic(std::shared_ptr<CartesianHMesh> mesh,
                                     ParticleGroupSharedPtr particle_group)
    : CartesianPeriodic(particle_group->sycl_target, mesh,
                        particle_group->position_dat) {}

CartesianPeriodic::CartesianPeriodic(std::shared_ptr<CartesianHMesh> mesh,
                                     ParticleGroupSharedPtr particle_group,
                                     std::vector<int> masks)
    : CartesianPeriodic(particle_group->sycl_target, mesh,
                        particle_group->position_dat, masks) {}

void CartesianPeriodic::execute() { this->pbc_loop->execute(); }

} // namespace NESO::Particles
