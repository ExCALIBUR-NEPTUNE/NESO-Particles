#ifndef _NESO_PARTICLES_ALGORITHMS_CELLWISE_METHODS_HPP_
#define _NESO_PARTICLES_ALGORITHMS_CELLWISE_METHODS_HPP_

#include "../compute_target.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

namespace NESO::Particles {

/**
 * Get the number of particles in each cell as a CellDatConst.
 *
 * @param particle_sub_group Particle{Sub}Group containing particles.
 * @param cell_dat_const CellDatConst to populate with particle counts.
 * @param row Row to populate in the CellDatConst.
 * @param col Column to populate in the CellDatConst.
 */
template <typename GROUP_TYPE, typename VALUE_TYPE>
void get_npart_cell(std::shared_ptr<GROUP_TYPE> particle_sub_group,
                    CellDatConstSharedPtr<VALUE_TYPE> cell_dat_const,
                    const int row = 0, const int col = 0) {

  auto particle_group = get_particle_group(particle_sub_group);
  const int cell_count = particle_group->domain->mesh->get_cell_count();

  NESOASSERT(cell_dat_const->ncells == cell_count,
             "Bad cell count in cell_dat_const");
  NESOASSERT(cell_dat_const->nrow > row, "Bad row passed: out of bounds.");
  NESOASSERT(cell_dat_const->ncol > col, "Bad col passed: out of bounds.");

  const int stride = cell_dat_const->nrow * cell_dat_const->ncol;
  const int nrow = cell_dat_const->nrow;
  auto sycl_target = particle_group->sycl_target;

  auto d_cell_dat_const_ptr = cell_dat_const->device_ptr();

  if constexpr (std::is_same<GROUP_TYPE, ParticleSubGroup>::value == true) {
    particle_sub_group->create_if_required();
    auto selection = particle_sub_group->get_selection();
    auto *k_npart_cell = selection.d_npart_cell;
    sycl_target->queue
        .parallel_for(sycl::range<1>(cell_count),
                      [=](auto idx) {
                        d_cell_dat_const_ptr[idx * stride + nrow * col + row] =
                            k_npart_cell[idx];
                      })
        .wait_and_throw();
  } else {
    auto *k_npart_cell = particle_group->cell_id_dat->d_npart_cell;
    sycl_target->queue
        .parallel_for(sycl::range<1>(cell_count),
                      [=](auto idx) {
                        d_cell_dat_const_ptr[idx * stride + nrow * col + row] =
                            k_npart_cell[idx];
                      })
        .wait_and_throw();
  }
}

extern template void get_npart_cell(std::shared_ptr<ParticleGroup>,
                                    CellDatConstSharedPtr<int>, int, int);
extern template void get_npart_cell(std::shared_ptr<ParticleGroup>,
                                    CellDatConstSharedPtr<INT>, int, int);
extern template void get_npart_cell(std::shared_ptr<ParticleSubGroup>,
                                    CellDatConstSharedPtr<int>, int, int);
extern template void get_npart_cell(std::shared_ptr<ParticleSubGroup>,
                                    CellDatConstSharedPtr<INT>, int, int);

} // namespace NESO::Particles

#endif
