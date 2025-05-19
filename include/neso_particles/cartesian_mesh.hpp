#ifndef _NESO_CARTESIAN_MESH
#define _NESO_CARTESIAN_MESH

#include <memory>
#include <mpi.h>

#include "cartesian_mesh/cartesian_cell_bin.hpp"
#include "compute_target.hpp"
#include "local_mapping.hpp"
#include "loop/particle_loop.hpp"
#include "particle_dat.hpp"
#include "particle_group.hpp"
#include "profiling.hpp"
#include "sycl_typedefs.hpp"
#include "typedefs.hpp"

namespace NESO::Particles {

/**
 *  LocalMapper for CartesianHMesh. Maps particle positions within the stencil
 *  width of each MPI subdomain to the owning rank.
 */
class CartesianHMeshLocalMapperT : public LocalMapper {
protected:
  SYCLTargetSharedPtr sycl_target;
  int ndim;
  REAL cell_width_fine;
  REAL inverse_cell_width_fine;

  // For each dimension map from cell to coord in cart comm
  std::unique_ptr<BufferDeviceHost<int>> dh_cell_lookups[3];
  // array to hold the pointers to each dh_cell_lookups dim
  std::unique_ptr<BufferDeviceHost<int *>> dh_cell_lookup;
  // Map from linear rank computed lexicographically to rank computed by the
  // cart comm from the dims.
  std::unique_ptr<BufferDeviceHost<int>> dh_rank_map;
  // The dims of the cart comm
  std::unique_ptr<BufferDeviceHost<int>> dh_dims;
  // Cell counts of the underlying mesh.
  std::unique_ptr<BufferDeviceHost<int>> dh_cell_counts;
  // Cell binning for cell move
  std::unique_ptr<CartesianCellBin> cart_cell_bin;

public:
  /// Disable (implicit) copies.
  CartesianHMeshLocalMapperT(const CartesianHMeshLocalMapperT &st) = delete;
  /// Disable (implicit) copies.
  CartesianHMeshLocalMapperT &
  operator=(CartesianHMeshLocalMapperT const &a) = delete;
  virtual ~CartesianHMeshLocalMapperT() = default;

  /// CartesianHMesh on which the lookup is based.
  CartesianHMeshSharedPtr mesh;
  /**
   *  Construct a new mapper instance to map local particle positions to owing
   * ranks.
   *
   *  @param sycl_target SYCLTargetSharedPtr to use as compute device.
   *  @param mesh CartesianHMesh instance this mapping is based on.
   */
  CartesianHMeshLocalMapperT(SYCLTargetSharedPtr sycl_target,
                             CartesianHMeshSharedPtr mesh);

  /**
   *  Map positions to owning MPI ranks. Positions should be within the domain
   *  prior to calling map, i.e. particles should be within the domain extents.
   *
   *  @param particle_group ParticleGroup to use.
   */
  void map(ParticleGroup &particle_group, const int map_cell = -1) override;

  /**
   *  Map positions to owning cells. Positions should be within the domain
   *  prior to calling map, i.e. particles should be within the domain extents.
   *
   *  @param particle_group ParticleGroup to use.
   *  @param map_cell Cell to map.
   */
  void map_cells(ParticleGroup &particle_group,
                 const int map_cell = -1) override;

  /**
   *  No-op implementation of callback.
   *
   *  @param particle_group ParticleGroup.
   */
  inline void particle_group_callback(
      [[maybe_unused]] ParticleGroup &particle_group) override {};
};

std::shared_ptr<CartesianHMeshLocalMapperT>
CartesianHMeshLocalMapper(SYCLTargetSharedPtr sycl_target,
                          CartesianHMeshSharedPtr mesh);

} // namespace NESO::Particles
#endif
