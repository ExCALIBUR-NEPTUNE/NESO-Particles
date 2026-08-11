#include <neso_particles/nd_host_array.hpp>
#include <neso_particles/nd_host_array_impl.hpp>

namespace NESO::Particles {

template class NDHostArray<REAL, 2>;
template class NDHostArray<INT, 2>;
template class NDHostArray<int, 2>;
template class NDHostArray<REAL, 3>;
template class NDHostArray<INT, 3>;
template class NDHostArray<int, 3>;

} // namespace NESO::Particles
