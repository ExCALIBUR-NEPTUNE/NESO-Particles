#ifndef _NESO_PARTICLES_UTILITY
#define _NESO_PARTICLES_UTILITY

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "access.hpp"
#include "compute_target.hpp"
#include "particle_spec.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

inline std::vector<std::vector<double>>
uniform_within_extents(const int N, const int ndim, const double *extents,
                       std::mt19937 rng = std::mt19937()) {

  std::uniform_real_distribution<double> uniform_rng(0.0, 1.0);
  std::vector<std::vector<double>> positions(ndim);
  for (int dimx = 0; dimx < ndim; dimx++) {
    positions[dimx] = std::vector<double>(N);
    const double ex = extents[dimx];
    for (int px = 0; px < N; px++) {
      positions[dimx][px] = ex * uniform_rng(rng);
    }
  }

  return positions;
}

} // namespace NESO::Particles

#endif
