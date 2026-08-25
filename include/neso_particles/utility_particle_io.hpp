#ifndef _NESO_PARTICLES_UTILITY_PARTICLE_IO_HPP_
#define _NESO_PARTICLES_UTILITY_PARTICLE_IO_HPP_

#include "external_interfaces/vtk/vtk.hpp"
#include "loop/particle_loop_functions.hpp"
#include "loop/particle_loop_impl.hpp"
#include "particle_group.hpp"
#include "particle_sub_group/particle_loop_sub_group_functions.hpp"
#include "particle_sub_group/particle_sub_group_utility.hpp"

namespace NESO::Particles {

/**
 * Get VTK representation of the trajectory between two particle positions. This
 * function is not efficient and is for debugging.
 *
 * @param[in] particle_sub_group Particle{Sub}Group of particles.
 * @param[in] sym_start Sym that determines start of the trajectory.
 * @param[in] sym_end Sym that determines end of the trajectory.
 * @param[in, out] trajectory Output trajectory.
 */
template <typename GROUP_TYPE>
inline void
get_vtk_trajectory_line(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                        Sym<REAL> sym_start, Sym<REAL> sym_end,
                        std::vector<VTK::UnstructuredCell> &trajectory) {

  auto particle_group = get_particle_group(particle_sub_group);
  auto sycl_target = particle_group->sycl_target;
  const auto npart_local = particle_sub_group->get_npart_local();

  NESOASSERT(particle_group->contains_dat(sym_start), "Bad sym_start.");
  NESOASSERT(particle_group->contains_dat(sym_end), "Bad sym_end.");

  const auto ncomp = particle_group->get_dat(sym_start)->ncomp;
  NESOASSERT(
      particle_group->get_dat(sym_end)->ncomp == ncomp,
      "Missmatch in number of components between sym_start and sym_end.");

  auto d_data = get_resource<BufferDevice<REAL>,
                             ResourceStackInterfaceBufferDevice<REAL>>(
      sycl_target->resource_stack_map, ResourceStackKeyBufferDevice<REAL>{},
      sycl_target);
  d_data->realloc_no_copy(2 * npart_local * ncomp);
  REAL *RESTRICT k_data = d_data->ptr;

  auto h_data =
      get_resource<BufferHost<REAL>, ResourceStackInterfaceBufferHost<REAL>>(
          sycl_target->resource_stack_map, ResourceStackKeyBufferHost<REAL>{},
          sycl_target);
  h_data->realloc_no_copy(2 * npart_local * ncomp);

  particle_loop(
      particle_sub_group,
      [=](auto INDEX, auto SYM_START, auto SYM_END) {
        const auto index = INDEX.get_loop_linear_index();
        for (int cx = 0; cx < ncomp; cx++) {
          // This ordering makes the output be start, end.
          k_data[(cx + ncomp) * npart_local + index] = SYM_START.at(cx);
          k_data[cx * npart_local + index] = SYM_END.at(cx);
        }
      },
      Access::read(ParticleLoopIndex{}), Access::read(sym_start),
      Access::read(sym_end))
      ->execute();

  auto e0 = sycl_target->queue.memcpy(h_data->ptr, k_data,
                                      npart_local * ncomp * 2 * sizeof(REAL));

  trajectory.clear();
  trajectory.resize(npart_local);
  e0.wait_and_throw();

  for (int px = 0; px < npart_local; px++) {
    trajectory[px].num_points = 2;
    trajectory[px].points.reserve(6);
    for (int vx = 0; vx < ncomp; vx++) {
      trajectory[px].points.push_back(h_data->ptr[vx * npart_local + px]);
    }
    for (int vx = ncomp; vx < 3; vx++) {
      trajectory[px].points.push_back(0.0);
    }
    for (int vx = 0; vx < ncomp; vx++) {
      trajectory[px].points.push_back(
          h_data->ptr[(vx + ncomp) * npart_local + px]);
    }
    for (int vx = ncomp; vx < 3; vx++) {
      trajectory[px].points.push_back(0.0);
    }
    trajectory[px].cell_type = VTK::CellType::line;
  }

  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferHost<REAL>{}, h_data);
  restore_resource(sycl_target->resource_stack_map,
                   ResourceStackKeyBufferDevice<REAL>{}, d_data);
}

extern template void get_vtk_trajectory_line<ParticleGroup>(
    std::shared_ptr<ParticleGroup> particle_sub_group, Sym<REAL> sym_start,
    Sym<REAL> sym_end, std::vector<VTK::UnstructuredCell> &trajectory);
extern template void get_vtk_trajectory_line<ParticleSubGroup>(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<REAL> sym_start,
    Sym<REAL> sym_end, std::vector<VTK::UnstructuredCell> &trajectory);

} // namespace NESO::Particles

#endif
