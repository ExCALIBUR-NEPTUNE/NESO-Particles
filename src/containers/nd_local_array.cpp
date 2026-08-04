#include <neso_particles/containers/nd_local_array.hpp>

namespace NESO::Particles {
template class NDLocalArray<REAL, 2>;
template class NDLocalArray<INT, 2>;
template class NDLocalArray<int, 2>;
template class NDLocalArray<REAL, 3>;
template class NDLocalArray<INT, 3>;
template class NDLocalArray<int, 3>;
} // namespace NESO::Particles
