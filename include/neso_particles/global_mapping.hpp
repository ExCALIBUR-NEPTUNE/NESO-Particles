#ifndef _NESO_PARTICLES_GLOBAL_MAPPING
#define _NESO_PARTICLES_GLOBAL_MAPPING

#include <CL/sycl.hpp>
#include <mpi.h>

#include "domain.hpp"
#include "error_propagate.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/**
 * This class maps global positions into the cells of a HMesh and determines
 * which MPI rank owns that cell.
 */
class MeshHierarchyGlobalMap {
private:
  /// Compute device used by the instance.
  SYCLTargetSharedPtr sycl_target;
  /// HMesh instance on which particles live.
  HMeshSharedPtr h_mesh;
  /// ParticleDat storing Positions
  ParticleDatSharedPtr<REAL> &position_dat;
  /// ParticleDat storing cell ids
  ParticleDatSharedPtr<INT> &cell_id_dat;
  /// ParticleDat storing MPI rank
  ParticleDatSharedPtr<INT> &mpi_rank_dat;

  /// Host buffer containing the number of particles to query owning MPI rank
  BufferHost<int> h_lookup_count;
  /// Device buffer containing the number of particles to query owning MPI rank
  BufferDevice<int> d_lookup_count;

  /// Host buffer of global cells on HMesh to query owner of.
  BufferHost<INT> h_lookup_global_cells;
  /// Device buffer of global cells on HMesh to query owner of.
  BufferDevice<INT> d_lookup_global_cells;
  /// Space to store the ranks owning the lookup cells on the host.
  BufferHost<int> h_lookup_ranks;
  /// Space to store the ranks owning the lookup cells on the Device.
  BufferDevice<int> d_lookup_ranks;

  /// Cells of the particles for which the lookup is being performed.
  BufferDevice<int> d_lookup_local_cells;
  /// Layers of the particles for which the lookup is being performed.
  BufferDevice<int> d_lookup_local_layers;

  /// Host buffer holding the origin of MeshHierarchy
  BufferHost<REAL> h_origin;
  /// Device buffer holding the origin of MeshHierarchy
  BufferDevice<REAL> d_origin;
  /// Host buffer containing the dims of MeshHierarchy
  BufferHost<int> h_dims;
  /// Device buffer containing the dims of MeshHierarchy
  BufferDevice<int> d_dims;

  /// ErrorPropagate for detecting errors in kernels
  ErrorPropagate error_propagate;

public:
  /// Disable (implicit) copies.
  MeshHierarchyGlobalMap(const MeshHierarchyGlobalMap &st) = delete;
  /// Disable (implicit) copies.
  MeshHierarchyGlobalMap &operator=(MeshHierarchyGlobalMap const &a) = delete;

  ~MeshHierarchyGlobalMap(){};

  /**
   * Construct a new global mapping instance for MeshHierarchy.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param h_mesh HMesh derived mesh to use for mapping.
   * @param position_dat ParticleDat containing particle positions.
   * @param cell_id_dat ParticleDat containg particle cell ids.
   * @param mpi_rank_dat ParticleDat containing the owning rank of particles.
   */
  MeshHierarchyGlobalMap(SYCLTargetSharedPtr sycl_target, HMeshSharedPtr h_mesh,
                         ParticleDatSharedPtr<REAL> &position_dat,
                         ParticleDatSharedPtr<INT> &cell_id_dat,
                         ParticleDatSharedPtr<INT> &mpi_rank_dat)
      : sycl_target(sycl_target), h_mesh(h_mesh), position_dat(position_dat),
        cell_id_dat(cell_id_dat), mpi_rank_dat(mpi_rank_dat),
        h_lookup_count(sycl_target, 1), d_lookup_count(sycl_target, 1),
        h_lookup_global_cells(sycl_target, 1),
        d_lookup_global_cells(sycl_target, 1), h_lookup_ranks(sycl_target, 1),
        d_lookup_ranks(sycl_target, 1), d_lookup_local_cells(sycl_target, 1),
        d_lookup_local_layers(sycl_target, 1), h_origin(sycl_target, 3),
        d_origin(sycl_target, 3), h_dims(sycl_target, 3),
        d_dims(sycl_target, 3), error_propagate(sycl_target){};

  /**
   * For each particle that does not have a non-negative MPI rank determined as
   * a local owner obtain the MPI rank that owns the global cell which contains
   * the particle.
   */
  inline void execute();
};

/**
 *  Set all components 0 and 1 of particles to -1 in the passed ParticleDat.
 *
 *  @param mpi_rank_dat ParticleDat containing MPI ranks to reset.
 */
inline void reset_mpi_ranks(ParticleDatSharedPtr<INT> mpi_rank_dat);

} // namespace NESO::Particles
#endif
