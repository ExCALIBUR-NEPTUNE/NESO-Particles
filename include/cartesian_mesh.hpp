#ifndef _NESO_CARTESIAN_MESH
#define _NESO_CARTESIAN_MESH

#include <CL/sycl.hpp>
#include <memory>
#include <mpi.h>

#include "compute_target.hpp"
#include "local_mapping.hpp"
#include "particle_dat.hpp"
#include "profiling.hpp"
#include "typedefs.hpp"

using namespace cl;
namespace NESO::Particles {

/*
 *  LocalMapper for CartesianHMesh
 */
class CartesianHMeshLocalMapperT : public LocalMapper {
private:
  SYCLTarget &sycl_target;
  BufferDeviceHost<int> dh_dims;
  int ndim;
  REAL cell_width_fine;
  REAL inverse_cell_width_fine;

public:
  BufferDeviceHost<int> dh_lookup;
  BufferDeviceHost<int> dh_lookup_dims;
  BufferDeviceHost<int> dh_map;
  int lookup_stride;

  CartesianHMesh &mesh;
  CartesianHMeshLocalMapperT(SYCLTarget &sycl_target, CartesianHMesh &mesh)
      : sycl_target(sycl_target), mesh(mesh), dh_dims(sycl_target, mesh.ndim),
        dh_map(sycl_target, 1), dh_lookup(sycl_target, 1),
        dh_lookup_dims(sycl_target, mesh.ndim) {
    this->ndim = mesh.ndim;

    this->lookup_stride = 0;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      const int cc = mesh.cell_counts[dimx];
      this->dh_dims.h_buffer.ptr[dimx] = cc;
      this->lookup_stride = MAX(cc, this->lookup_stride);
    }
    this->dh_dims.host_to_device();
    this->cell_width_fine = mesh.get_cell_width_fine();
    this->inverse_cell_width_fine = mesh.get_inverse_cell_width_fine();

    this->dh_lookup.realloc_no_copy(this->ndim * this->lookup_stride);

    // provide a value to indicate this rank does not own this cell in each
    // dimension
    for (int cx = 0; cx < this->ndim * this->lookup_stride; cx++) {
      this->dh_lookup.h_buffer.ptr[cx] = -1;
    }

    int map_size = 1;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      map_size *= mesh.cell_counts_local[dimx] + 2 * mesh.stencil_width;
    }
    this->dh_map.realloc_no_copy(map_size);
    for (int mx = 0; mx < map_size; mx++) {
      this->dh_map.h_buffer.ptr[mx] = -1;
    }

    // populate which cells are owned in each dimension
    const auto stencil_width = this->mesh.stencil_width;
    for (int dimx = 0; dimx < this->ndim; dimx++) {
      int index = 0;
      const int start = this->mesh.cell_starts[dimx] - stencil_width;
      const int end = this->mesh.cell_ends[dimx] + stencil_width;
      auto cc = mesh.cell_counts[dimx];
      for (int cellx = start; cellx < end; cellx++) {
        // wrap the cell into the domain
        const int cell_wrapped = (cellx + stencil_width * cc) % cc;
        this->dh_lookup.h_buffer
            .ptr[dimx * this->lookup_stride + cell_wrapped] = index;
        index++;
      }
      this->dh_lookup_dims.h_buffer.ptr[dimx] = index;
      NESOASSERT(index == mesh.cell_counts_local[dimx] + 2 * stencil_width,
                 "Bad index value");
    }
    this->dh_lookup.host_to_device();
    this->dh_lookup_dims.host_to_device();

    // construct the map for the locally owned cells

    int starts[3] = {0, 0, 0};
    int ends[3] = {1, 1, 1};
    int cell_counts[3] = {1, 1, 1};

    for (int dimx = 0; dimx < this->ndim; dimx++) {
      starts[dimx] = this->mesh.cell_starts[dimx] - stencil_width;
      ends[dimx] = this->mesh.cell_ends[dimx] + stencil_width;
      cell_counts[dimx] = this->mesh.cell_counts[dimx];
    }

    // mesh tuple index
    INT index_mesh[3];
    // mesh_hierarchy tuple index
    INT index_mh[6];

    auto h_lookup_x = this->dh_lookup.h_buffer.ptr;
    auto h_lookup_y = this->dh_lookup.h_buffer.ptr + this->lookup_stride;
    auto h_lookup_z = this->dh_lookup.h_buffer.ptr + this->lookup_stride * 2;

    auto stride_x = this->dh_lookup_dims.h_buffer.ptr[0];
    auto stride_y = this->dh_lookup_dims.h_buffer.ptr[1];

    for (int cz = starts[2]; cz < ends[2]; cz++) {
      const int czs = (cz + stencil_width * cell_counts[2]) % cell_counts[2];
      index_mesh[2] = czs;
      const int local_tuple_z = (this->ndim > 2) ? h_lookup_z[czs] : 0;
      for (int cy = starts[1]; cy < ends[1]; cy++) {
        const int cys = (cy + stencil_width * cell_counts[1]) % cell_counts[1];
        index_mesh[1] = cys;
        const int local_tuple_y = (this->ndim > 1) ? h_lookup_y[cys] : 0;
        for (int cx = starts[0]; cx < ends[0]; cx++) {
          const int cxs =
              (cx + stencil_width * cell_counts[0]) % cell_counts[0];
          index_mesh[0] = cxs;

          const int local_tuple_x = h_lookup_x[cxs];
          // convert mesh tuple index to mesh hierarchy tuple index
          this->mesh.mesh_tuple_to_mh_tuple(index_mesh, index_mh);
          // convert mesh hierarchy tuple index to global linear index in the
          // MeshHierarchy
          const INT index_global =
              this->mesh.get_mesh_hierarchy()->tuple_to_linear_global(index_mh);
          // get the rank that owns that global cell
          const int remote_rank =
              this->mesh.get_mesh_hierarchy()->get_owner(index_global);

          const int linear_index =
              local_tuple_x +
              stride_x * (local_tuple_y + stride_y * local_tuple_z);

          this->dh_map.h_buffer.ptr[linear_index] = remote_rank;
        }
      }
    }

    this->dh_map.host_to_device();
  };

  inline void map(ParticleDatShPtr<REAL> &position_dat,
                  ParticleDatShPtr<INT> &cell_id_dat,
                  ParticleDatShPtr<INT> &mpi_rank_dat) {

    // pointers to access dats in kernel
    auto k_position_dat = position_dat->cell_dat.device_ptr();
    auto k_mpi_rank_dat = mpi_rank_dat->cell_dat.device_ptr();

    auto k_ndim = this->ndim;
    auto k_dims = this->dh_dims.d_buffer.ptr;
    auto k_cell_width_fine = this->cell_width_fine;
    auto k_inverse_cell_width_fine = this->inverse_cell_width_fine;
    auto k_lookup = this->dh_lookup.d_buffer.ptr;
    auto k_lookup_stride = this->lookup_stride;
    auto k_lookup_dims = this->dh_lookup_dims.d_buffer.ptr;
    auto k_map = this->dh_map.d_buffer.ptr;

    // iteration set
    auto pl_iter_range = mpi_rank_dat->get_particle_loop_iter_range();
    auto pl_stride = mpi_rank_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = mpi_rank_dat->get_particle_loop_npart_cell();

    this->sycl_target.queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                int local_tuple[3] = {0, 0, 0};
                int mask = 0;
                // k_mpi_rank_dat[cellx][1][layerx];
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  const REAL pos = k_position_dat[cellx][dimx][layerx];
                  int cell_fine = ((REAL)pos * k_inverse_cell_width_fine);
                  if (cell_fine >= k_dims[dimx]) {
                    cell_fine = k_dims[dimx] - 1;
                  } else if (cell_fine < 0) {
                    cell_fine = 0;
                  }
                  // use the cell to get an index for the dimension
                  const int dim_index =
                      k_lookup[k_lookup_stride * dimx + cell_fine];
                  local_tuple[dimx] = dim_index;

                  // if the mask takes a negative value then a dim_index had a
                  // negative value
                  // => cell is not in the local map
                  mask = MIN(mask, dim_index);
                }

                const int linear_index =
                    local_tuple[0] +
                    k_lookup_dims[0] *
                        (local_tuple[1] + k_lookup_dims[1] * local_tuple[2]);

                const int remote_rank = (mask < 0) ? -1 : k_map[linear_index];
                k_mpi_rank_dat[cellx][1][layerx] = remote_rank;

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
  };
};

inline std::shared_ptr<CartesianHMeshLocalMapperT>
CartesianHMeshLocalMapper(SYCLTarget &sycl_target, CartesianHMesh &mesh) {
  return std::make_shared<CartesianHMeshLocalMapperT>(sycl_target, mesh);
}

} // namespace NESO::Particles
#endif
