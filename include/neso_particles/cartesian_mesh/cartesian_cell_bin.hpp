#ifndef _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_CELL_BIN_HPP_
#define _NESO_PARTICLES_CARTESIAN_MESH_CARTESIAN_CELL_BIN_HPP_

#include <cmath>

#include "../domain.hpp"
#include "../loop/particle_loop.hpp"
#include "../particle_dat.hpp"
#include "../profiling.hpp"
#include "../sycl_typedefs.hpp"
#include "../typedefs.hpp"
#include "cartesian_h_mesh.hpp"

namespace NESO::Particles {

/**
 * Bin particle positions into the cells of a CartesianHMesh.
 */
class CartesianCellBin {
protected:
  BufferDevice<int> d_cell_counts;
  BufferDevice<int> d_cell_starts;
  BufferDevice<int> d_cell_ends;

  SYCLTargetSharedPtr sycl_target;
  CartesianHMeshSharedPtr mesh;
  ParticleDatSharedPtr<REAL> position_dat;
  ParticleDatSharedPtr<INT> cell_id_dat;

  ParticleLoopSharedPtr get_loop(ParticleDatSharedPtr<REAL> position_dat,
                                 ParticleDatSharedPtr<INT> cell_id_dat);

public:
  /// Disable (implicit) copies.
  CartesianCellBin(const CartesianCellBin &st) = delete;
  /// Disable (implicit) copies.
  CartesianCellBin &operator=(CartesianCellBin const &a) = delete;

  ~CartesianCellBin(){};

  /**
   * Create instance to bin particles into cells of a CartesianHMesh.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param mesh CartesianHMeshSharedPtr to containing the particles.
   * @param position_dat ParticleDat with components equal to the mesh dimension
   * containing particle positions.
   * @param cell_id_dat ParticleDat to write particle cell ids to.
   */
  CartesianCellBin(SYCLTargetSharedPtr sycl_target,
                   CartesianHMeshSharedPtr mesh);

  /**
   * Create instance to bin particles into cells of a CartesianHMesh.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param mesh CartesianHMeshSharedPtr to containing the particles.
   * @param position_dat ParticleDat with components equal to the mesh dimension
   * containing particle positions.
   * @param cell_id_dat ParticleDat to write particle cell ids to.
   */
  CartesianCellBin(SYCLTargetSharedPtr sycl_target,
                   CartesianHMeshSharedPtr mesh,
                   ParticleDatSharedPtr<REAL> position_dat,
                   ParticleDatSharedPtr<INT> cell_id_dat);

  /**
   *  Apply the cell binning kernel to each particle stored on this MPI rank.
   *  Particles must be within the domain region owned by this MPI rank.
   */
  void execute();

  /**
   * Map call for LocalMapper.
   *
   *  @param particle_group ParticleGroup to use.
   *  @param map_cell Cell to map.
   */
  void map_cells(ParticleGroup &particle_group,
                 [[maybe_unused]] const int map_cell = -1);
};

} // namespace NESO::Particles

#endif
