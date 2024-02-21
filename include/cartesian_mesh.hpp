#ifndef _NESO_CARTESIAN_MESH
#define _NESO_CARTESIAN_MESH

#include <CL/sycl.hpp>
#include <memory>
#include <mpi.h>

#include "compute_target.hpp"
#include "local_mapping.hpp"
#include "loop/particle_loop.hpp"
#include "particle_dat.hpp"
#include "particle_group.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
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

public:
  /// Disable (implicit) copies.
  CartesianHMeshLocalMapperT(const CartesianHMeshLocalMapperT &st) = delete;
  /// Disable (implicit) copies.
  CartesianHMeshLocalMapperT &
  operator=(CartesianHMeshLocalMapperT const &a) = delete;

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
                             CartesianHMeshSharedPtr mesh)
      : sycl_target(sycl_target), mesh(mesh) {

    this->ndim = mesh->ndim;
    this->cell_width_fine = mesh->get_cell_width_fine();
    this->inverse_cell_width_fine = mesh->get_inverse_cell_width_fine();

    for (int dimx = 0; dimx < ndim; dimx++) {
      this->dh_cell_lookups[dimx] = std::make_unique<BufferDeviceHost<int>>(
          sycl_target, mesh->cell_counts[dimx]);
      for (int cx = 0; cx < mesh->cell_counts[dimx]; cx++) {
        this->dh_cell_lookups[dimx]->h_buffer.ptr[cx] = -1;
      }
    }

    // build the maps in each dimension
    int starts[3] = {0, 0, 0};
    int ends[3] = {1, 1, 1};
    int cell_counts[3] = {1, 1, 1};
    // compute start end cells that are owned by this cell or within the halo

    this->dh_cell_counts =
        std::make_unique<BufferDeviceHost<int>>(sycl_target, 3);
    const auto stencil_width = this->mesh->stencil_width;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      starts[dimx] = this->mesh->cell_starts[dimx] - stencil_width;
      ends[dimx] = this->mesh->cell_ends[dimx] + stencil_width;
      cell_counts[dimx] = this->mesh->cell_counts[dimx];
    }
    for (int dimx = 0; dimx < 3; dimx++) {
      this->dh_cell_counts->h_buffer.ptr[dimx] = cell_counts[dimx];
    }
    this->dh_cell_counts->host_to_device();

    MPI_Comm comm = this->mesh->get_comm();
    int rank, size;
    MPICHK(MPI_Comm_rank(comm, &rank));
    MPICHK(MPI_Comm_size(comm, &size));

    for (int dimx = 0; dimx < this->ndim; dimx++) {
      for (int cx = starts[dimx]; cx < ends[dimx]; cx++) {
        // periodically map the cell index back into the domain
        const int cxs =
            (cx + stencil_width * cell_counts[dimx]) % cell_counts[dimx];
        // find an owner for this dimension
        // mesh tuple index
        INT index_mesh[3] = {0, 0, 0};
        // mesh_hierarchy tuple index
        INT index_mh[6] = {0, 0, 0, 0, 0, 0};

        // the structure of the cart comm means we can get the owning coord
        // with the other dimensions = 0
        index_mesh[dimx] = cxs;
        // convert mesh tuple index to mesh hierarchy tuple index
        this->mesh->mesh_tuple_to_mh_tuple(index_mesh, index_mh);
        // convert mesh hierarchy tuple index to global linear index in the
        // MeshHierarchy
        const INT index_global =
            this->mesh->get_mesh_hierarchy()->tuple_to_linear_global(index_mh);
        // get the rank that owns that global cell
        const int remote_rank =
            this->mesh->get_mesh_hierarchy()->get_owner(index_global);

        // get the cart comm coords for this MPI rank to get the coord in dimx
        int coords[3];
        MPICHK(MPI_Cart_coords(comm, remote_rank, this->ndim, coords));
        const int coord = coords[dimx];

        this->dh_cell_lookups[dimx]->h_buffer.ptr[cxs] = coord;
      }
    }
    this->dh_cell_lookup =
        std::make_unique<BufferDeviceHost<int *>>(sycl_target, 3);
    for (int dimx = 0; dimx < ndim; dimx++) {
      this->dh_cell_lookups[dimx]->host_to_device();
      this->dh_cell_lookup->h_buffer.ptr[dimx] =
          this->dh_cell_lookups[dimx]->d_buffer.ptr;
    }
    this->dh_cell_lookup->host_to_device();

    // build the map from lexicographic linearisation to the MPI implementation
    // linearisation. On most MPI implementations this will be an identity map.
    this->dh_rank_map =
        std::make_unique<BufferDeviceHost<int>>(sycl_target, size);
    for (int cx = 0; cx < size; cx++) {
      this->dh_rank_map->h_buffer.ptr[cx] = -1;
    }
    int dims[3] = {1, 1, 1};
    int periods[3];
    int coords[3];
    MPICHK(MPI_Cart_get(comm, this->ndim, dims, periods, coords));
    this->dh_dims = std::make_unique<BufferDeviceHost<int>>(sycl_target, 3);
    for (int dx = 0; dx < 3; dx++) {
      this->dh_dims->h_buffer.ptr[dx] = dims[dx];
    }
    this->dh_dims->host_to_device();
    for (int d0 = 0; d0 < dims[0]; d0++) {
      coords[0] = d0;
      for (int d1 = 0; d1 < dims[1]; d1++) {
        coords[1] = d1;
        for (int d2 = 0; d2 < dims[2]; d2++) {
          coords[2] = d2;
          int rank_impl;
          MPICHK(MPI_Cart_rank(comm, coords, &rank_impl));
          const int rank_linear = d0 + d1 * dims[0] + d2 * dims[0] * dims[1];
          NESOASSERT((rank_linear >= 0) && (rank_linear < size),
                     "bad linear mpi rank");
          NESOASSERT((rank_impl >= 0) && (rank_impl < size),
                     "bad implementation mpi rank");
          this->dh_rank_map->h_buffer.ptr[rank_linear] = rank_impl;
        }
      }
    }
    for (int cx = 0; cx < size; cx++) {
      NESOASSERT(this->dh_rank_map->h_buffer.ptr[cx] >= 0,
                 "rank map incorrectly built");
    }
    this->dh_rank_map->host_to_device();

    // self testing
    // mesh tuple index
    INT index_mesh[3];
    // mesh_hierarchy tuple index
    INT index_mh[6];
    for (int cz = starts[2]; cz < ends[2]; cz++) {
      const int czs = (cz + stencil_width * cell_counts[2]) % cell_counts[2];
      index_mesh[2] = czs;
      for (int cy = starts[1]; cy < ends[1]; cy++) {
        const int cys = (cy + stencil_width * cell_counts[1]) % cell_counts[1];
        index_mesh[1] = cys;
        for (int cx = starts[0]; cx < ends[0]; cx++) {
          const int cxs =
              (cx + stencil_width * cell_counts[0]) % cell_counts[0];
          index_mesh[0] = cxs;

          // convert mesh tuple index to mesh hierarchy tuple index
          this->mesh->mesh_tuple_to_mh_tuple(index_mesh, index_mh);
          // convert mesh hierarchy tuple index to global linear index in the
          // MeshHierarchy
          const INT index_global =
              this->mesh->get_mesh_hierarchy()->tuple_to_linear_global(
                  index_mh);
          // get the rank that owns that global cell
          const int remote_rank =
              this->mesh->get_mesh_hierarchy()->get_owner(index_global);

          int coord[3] = {0, 0, 0};
          coord[0] = this->dh_cell_lookups[0]->h_buffer.ptr[cxs];
          coord[1] = (this->ndim > 1)
                         ? this->dh_cell_lookups[1]->h_buffer.ptr[cys]
                         : 0;
          coord[2] = (this->ndim > 2)
                         ? this->dh_cell_lookups[2]->h_buffer.ptr[czs]
                         : 0;
          int *dims_tmp = this->dh_dims->h_buffer.ptr;
          const int rank_lex = coord[0] + coord[1] * dims_tmp[0] +
                               coord[2] * dims_tmp[0] * dims_tmp[1];
          const int rank_impl = this->dh_rank_map->h_buffer.ptr[rank_lex];

          NESOASSERT(rank_impl == remote_rank, "rank self test failure");
          auto neighbours = mesh->get_local_communication_neighbours();
          if (rank_impl != rank) {
            NESOASSERT(
                std::count(neighbours.begin(), neighbours.end(), rank_impl),
                "halo cell is not listed as a neighbour");
          }
        }
      }
    }
  };

  /**
   *  Map positions to owning MPI ranks. Positions should be within the domain
   *  prior to calling map, i.e. particles should be within the domain extents.
   *
   *  @param particle_group ParticleGroup to use.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1) {

    ParticleDatSharedPtr<REAL> &position_dat = particle_group.position_dat;
    ParticleDatSharedPtr<INT> &mpi_rank_dat = particle_group.mpi_rank_dat;

    auto k_ndim = this->ndim;
    auto k_cell_counts = this->dh_cell_counts->d_buffer.ptr;
    auto k_dims = this->dh_dims->d_buffer.ptr;
    auto k_cell_lookup = this->dh_cell_lookup->d_buffer.ptr;
    auto k_inverse_cell_width_fine = this->inverse_cell_width_fine;
    auto k_rank_map = this->dh_rank_map->d_buffer.ptr;

    ParticleLoop pl(
        "CartesianHMeshLocalMapper", position_dat,
        [=](auto position, auto mpi_rank) {
          if (mpi_rank[1] < 0) {
            int coords[3] = {0, 0, 0};

            for (int dimx = 0; dimx < k_ndim; dimx++) {
              const REAL pos = position[dimx];
              int cell_fine = ((REAL)pos * k_inverse_cell_width_fine);
              if (cell_fine >= k_cell_counts[dimx]) {
                cell_fine = k_cell_counts[dimx] - 1;
              } else if (cell_fine < 0) {
                cell_fine = 0;
              }
              const int dim_index = k_cell_lookup[dimx][cell_fine];
              coords[dimx] = dim_index;
            }
            const int rank_linear = coords[0] + coords[1] * k_dims[0] +
                                    coords[2] * k_dims[0] * k_dims[1];

            const bool cell_found =
                (coords[0] >= 0) && (coords[1] >= 0) && (coords[2] >= 0);
            const int remote_rank = cell_found ? k_rank_map[rank_linear] : -1;

            mpi_rank[1] = remote_rank;
          }
        },
        Access::read(position_dat), Access::write(mpi_rank_dat));

    if (map_cell == -1) {
      pl.execute();
    } else {
      pl.execute(map_cell);
    }
  };

  /**
   *  No-op implementation of callback.
   *
   *  @param particle_group ParticleGroup.
   */
  inline void particle_group_callback(ParticleGroup &particle_group){};
};

inline std::shared_ptr<CartesianHMeshLocalMapperT>
CartesianHMeshLocalMapper(SYCLTargetSharedPtr sycl_target,
                          CartesianHMeshSharedPtr mesh) {
  return std::make_shared<CartesianHMeshLocalMapperT>(sycl_target, mesh);
}

} // namespace NESO::Particles
#endif
