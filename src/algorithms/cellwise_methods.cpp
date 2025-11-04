#include <neso_particles/algorithms/cellwise_methods.hpp>

namespace NESO::Particles {

template void get_npart_cell(std::shared_ptr<ParticleGroup>,
                             CellDatConstSharedPtr<int>, int, int);
template void get_npart_cell(std::shared_ptr<ParticleGroup>,
                             CellDatConstSharedPtr<INT>, int, int);
template void get_npart_cell(std::shared_ptr<ParticleSubGroup>,
                             CellDatConstSharedPtr<int>, int, int);
template void get_npart_cell(std::shared_ptr<ParticleSubGroup>,
                             CellDatConstSharedPtr<INT>, int, int);

} // namespace NESO::Particles
