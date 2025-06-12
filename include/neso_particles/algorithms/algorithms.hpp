#ifndef _NESO_PARTICLES_ALGORITHMS_ALGORITHMS_HPP_
#define _NESO_PARTICLES_ALGORITHMS_ALGORITHMS_HPP_

#include "../compute_target.hpp"
#include "../particle_group.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"
#include "../containers/cell_dat_const.hpp"

namespace NESO::Particles {


namespace Kernel {
template<typename T> using plus = sycl::plus<T>;
}

template<typename T, typename OP>
inline void reduce_dat_component_cellwise(
  ParticleGroupSharedPtr particle_group,
  Sym<T> sym,
  const int sym_component,
  CellDatConstSharedPtr<T> cell_dat_const,
  const int cell_dat_const_row,
  const int cell_dat_const_col,
  OP op
){





}








}

#endif
