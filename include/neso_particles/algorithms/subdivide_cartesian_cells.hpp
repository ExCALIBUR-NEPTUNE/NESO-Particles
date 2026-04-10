#ifndef _NESO_PARTICLES_ALGORITHMS_SUBDIVIDE_CARTESIAN_CELLS_HPP_
#define _NESO_PARTICLES_ALGORITHMS_SUBDIVIDE_CARTESIAN_CELLS_HPP_

#include "../cartesian_mesh/cartesian_h_mesh.hpp"
#include "../compute_target.hpp"
#include "../containers/cell_dat.hpp"
#include "../particle_spec.hpp"
#include "../particle_sub_group/particle_sub_group.hpp"

#include <vector>

namespace NESO::Particles {

/**
 * Implementation to bin particles into subdivision of CartesianHMesh cells. For
 * each mesh cell D subdivisions are performed in each dimension. The subdivided
 * cells are linearly indexed lexicographically from fastest to slowest.
 */
class SubdivideCartesianCells {
protected:
  std::shared_ptr<BufferDevice<int>> d_num_subdivisions;
  std::shared_ptr<BufferDevice<REAL>> d_subdvision_inverse_widths;
  std::shared_ptr<BufferDevice<REAL>> d_origins;

public:
  /// Compute device.
  SYCLTargetSharedPtr sycl_target;

  /// The mesh on which the subdivisions are defined.
  CartesianHMeshSharedPtr mesh;

  /// The number of subdivisions per dimension in each cell.
  std::vector<int> num_subdivisions;

  /**
   * Create new instance with provided mesh and number of subdivisions.
   *
   * @param sycl_target Compute device.
   * @param mesh CartesianHMesh instance to subdivide cells of.
   * @param num_subdivisions Vector containing the number of subdivisions for
   * each mesh cell.
   */
  SubdivideCartesianCells(SYCLTargetSharedPtr sycl_target,
                          CartesianHMeshSharedPtr mesh,
                          std::vector<int> &num_subdivisions);

  /**
   * Map particles to subdivided cells.
   *
   * @param particle_sub_group Particle{Sub}Group of particles.
   * @param sym_name Output Sym in which to store subdivision cell.
   * @param sym_component Output component in which to store subdivision cell.
   */
  template <typename GROUP_TYPE>
  void map(std::shared_ptr<GROUP_TYPE> particle_sub_group, Sym<INT> sym_name,
           const int sym_component) {}
};

extern template void
SubdivideCartesianCells::map(std::shared_ptr<ParticleGroup> particle_sub_group,
                             Sym<INT> sym_name, const int sym_component);

extern template void SubdivideCartesianCells::map(
    std::shared_ptr<ParticleSubGroup> particle_sub_group, Sym<INT> sym_name,
    const int sym_component);

} // namespace NESO::Particles

#endif
