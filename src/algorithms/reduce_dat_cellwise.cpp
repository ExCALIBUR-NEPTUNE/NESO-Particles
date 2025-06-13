#include <neso_particles/algorithms/reduce_dat_cellwise.hpp>

namespace NESO::Particles {

template void reduce_dat_component_cellwise(
    ParticleGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int sym_component, CellDatConstSharedPtr<REAL> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<REAL> op);

template void reduce_dat_component_cellwise(
    ParticleGroupSharedPtr particle_sub_group, Sym<INT> sym,
    const int sym_component, CellDatConstSharedPtr<INT> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<INT> op);

template void reduce_dat_component_cellwise(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<REAL> sym,
    const int sym_component, CellDatConstSharedPtr<REAL> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<REAL> op);

template void reduce_dat_component_cellwise(
    ParticleSubGroupSharedPtr particle_sub_group, Sym<INT> sym,
    const int sym_component, CellDatConstSharedPtr<INT> cell_dat_const,
    const int cell_dat_const_row, const int cell_dat_const_col,
    Kernel::plus<INT> op);

template void reduce_dat_components_cellwise(
    ParticleGroupSharedPtr particle_group, Sym<REAL> sym,
    CellDatConstSharedPtr<REAL> cell_dat_const, Kernel::plus<REAL> op);

template void reduce_dat_components_cellwise(
    ParticleGroupSharedPtr particle_group, Sym<INT> sym,
    CellDatConstSharedPtr<INT> cell_dat_const, Kernel::plus<INT> op);

} // namespace NESO::Particles
